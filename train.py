import os
import shutil
import time
import datetime
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
import matplotlib.pyplot as plt

from dataloader import ValidDataset, RandomTrainDataset  # Ensure these are updated
from utils import AverageMeter, Loss_PSNR, save_checkpoint, VGGPerceptualLoss
from pytorch_ssim import SSIM
from arch import FW_SAT


# Load configuration
def load_config(config_path='option.yaml'):
    with open(config_path, 'r') as config_file:
        return yaml.safe_load(config_file)


def setup_environment(config):
    os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
    os.environ["CUDA_VISIBLE_DEVICES"] = config['gpu_id']
    os.makedirs(config['outf'], exist_ok=True)
    shutil.copy2('option.yaml', config['outf'])


def initialize_tensorboard(outf):
    return SummaryWriter(outf)


def load_datasets(config):
    """
    Load the training and validation datasets.
    """
    train_rgb_dir = 'flir/images_rgb_train/data'
    train_thermal_dir = 'flir/images_thermal_train/data'
    val_rgb_dir = 'flir/images_rgb_val/data'
    val_thermal_dir = 'flir/images_thermal_val/data'

    # Create datasets
    train_data = RandomTrainDataset(
        rgb_dir=train_rgb_dir,
        thermal_dir=train_thermal_dir,
        transform=None,  # No cropping or augmentation applied
    )
    val_data = ValidDataset(
        rgb_dir=val_rgb_dir,
        thermal_dir=val_thermal_dir,
        transform=None,  # Ensure this aligns with the validation logic
    )

    # Debugging: Check dataset sizes
    print(f"Train Dataset: {len(train_data)} samples")
    print(f"Validation Dataset: {len(val_data)} samples")

    # Ensure datasets are non-empty
    if len(train_data) == 0:
        raise ValueError("Training dataset is empty. Check dataset paths or files.")
    if len(val_data) == 0:
        raise ValueError("Validation dataset is empty. Check dataset paths or files.")

    # Create dataloaders
    train_loader = DataLoader(
        dataset=train_data,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_data,
        batch_size=1,  # Validation is done one sample at a time
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return train_loader, val_loader


def initialize_model():
    """
    Initialize the FW-SAT model.
    """
    model = FW_SAT().cuda()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    return model


def load_checkpoint(model, optimizer, resume_file):
    if os.path.isfile(resume_file):
        print(f"=> Loading checkpoint '{resume_file}'")
        checkpoint = torch.load(resume_file)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        return checkpoint['epoch'], checkpoint['iter']
    return 0, 0


def train_one_iteration(lr, rgb, hr, model, optimizer, criterion_L1, criterion_SSIM, criterion_Perceptual):
    """
    Train the model for one iteration.
    """
    lr, rgb, hr = lr.cuda(), rgb.cuda(), hr.cuda()

    optimizer.zero_grad()
    output = model(rgb, lr)

    l1_loss = criterion_L1(output, hr)
    ssim_loss = 1 - criterion_SSIM(output, hr)
    perceptual_loss = criterion_Perceptual(output, hr)
    loss = 7 * l1_loss + ssim_loss + 0.15 * perceptual_loss

    loss.backward()
    optimizer.step()
    return loss, l1_loss, ssim_loss, perceptual_loss


def validate(model, val_loader, criterion_PSNR, criterion_SSIM):
    """
    Validate the model on the validation dataset.
    """
    model.eval()
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()

    with torch.no_grad():
        for rgb, thermal in val_loader:
            rgb, thermal = rgb.cuda(), thermal.cuda()
            output = model(rgb)

            psnr = criterion_PSNR(output, thermal)
            ssim = criterion_SSIM(output, thermal)

            psnr_meter.update(psnr.item(), n=1)
            ssim_meter.update(ssim.item(), n=1)

    return psnr_meter.avg, ssim_meter.avg


def main():
    config = load_config()
    setup_environment(config)
    writer = initialize_tensorboard(config['outf'])

    print("\nLoading dataset...")
    train_loader, val_loader = load_datasets(config)

    # Loss functions
    criterion_L1 = nn.L1Loss().cuda()
    criterion_PSNR = Loss_PSNR().cuda()
    criterion_SSIM = SSIM().cuda()
    criterion_Perceptual = VGGPerceptualLoss().cuda()

    # Initialize model and optimizer
    model = initialize_model()
    optimizer = optim.Adam(model.parameters(), lr=float(config['init_lr']), betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000 * config['end_epoch'], eta_min=1e-6)

    start_epoch, iteration = load_checkpoint(model, optimizer, config['resume_file'])
    total_iteration = 1000 * config['end_epoch']

    prev_time = time.time()

    # Training loop
    while iteration < total_iteration:
        model.train()
        for lr, rgb, hr in train_loader:
            loss, l1_loss, ssim_loss, perceptual_loss = train_one_iteration(
                lr, rgb, hr, model, optimizer, criterion_L1, criterion_SSIM, criterion_Perceptual
            )

            iteration += 1

            if iteration % 100 == 0:
                print(f'[Iter: {iteration}/{total_iteration}], LR={optimizer.param_groups[0]["lr"]:.9f}, '
                      f'Train Loss={loss:.9f}, L1 Loss={l1_loss:.9f}, SSIM Loss={ssim_loss:.9f}, Perceptual Loss={perceptual_loss:.9f}')

            if iteration % (1000 * (16 // config['batch_size'])) == 0:
                psnr, ssim = validate(model, val_loader, criterion_PSNR, criterion_SSIM)
                print(f'[Epoch: {iteration // 1000}/{config["end_epoch"]}], PSNR={psnr:.4f}, SSIM={ssim:.4f}')
                writer.add_scalar('PSNR', psnr, iteration // 1000)
                writer.add_scalar('SSIM', ssim, iteration // 1000)

            time_left = datetime.timedelta(seconds=(total_iteration - iteration) * (time.time() - prev_time))
            prev_time = time.time()
            print(f'Time left: {time_left}')

    print("Training complete.")


if __name__ == '__main__':
    main()


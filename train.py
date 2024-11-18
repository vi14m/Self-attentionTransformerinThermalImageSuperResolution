import os
import cv2
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

from dataloader import ValidDataset, RandomTrainDataset
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
    train_data = RandomTrainDataset(crop_size=config['patch_size'], upscale=config['upscale_factor'])
    val_data = ValidDataset(upscale=config['upscale_factor'])
    train_loader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(dataset=val_data, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)
    return train_loader, val_loader


def initialize_model():
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


def plot_loss(train_losses, val_losses, epoch):
    """Plots training and validation loss over epochs."""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training vs Validation Loss at Epoch {epoch}')
    plt.legend()
    plt.show()


def calculate_metrics(model, val_loader, criterion_PSNR, criterion_SSIM):
    """Calculates PSNR and SSIM for the validation set."""
    model.eval()
    psnr_values = []
    ssim_values = []
    with torch.no_grad():
        for lr, rgb, hr in val_loader:
            lr, rgb, hr = lr.cuda(), rgb.cuda(), hr.cuda()
            output = model(rgb, lr)
            psnr = criterion_PSNR(output, hr)
            ssim = criterion_SSIM(output, hr)
            psnr_values.append(psnr.item())
            ssim_values.append(ssim.item())
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    return avg_psnr, avg_ssim


def log_images(writer, iteration, lr, rgb, output, hr):
    """Logs input and output images to TensorBoard."""
    # Log sample images for inspection
    writer.add_image('Low-Resolution Image', lr[0].cpu(), iteration)
    writer.add_image('Ground Truth Image', hr[0].cpu(), iteration)
    writer.add_image('Generated Image', output[0].cpu(), iteration)
    print(f'Logged images for iteration {iteration}')


class EarlyStopping:
    """Early stopping to stop training if validation loss stops improving."""
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop_counter = 0

    def step(self, val_loss):
        """Checks if the validation loss has improved."""
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss < self.best_score - self.delta:
            self.best_score = val_loss
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1

        if self.early_stop_counter >= self.patience:
            print(f'Early stopping triggered after {self.patience} epochs with no improvement.')
            return True
        return False


def validate(model, val_loader, criterion_PSNR, criterion_SSIM):
    model.eval()
    losses_psnr = AverageMeter()
    losses_ssim = AverageMeter()

    for lr, rgb, hr in val_loader:
        lr, rgb, hr = lr.cuda(), rgb.cuda(), hr.cuda()
        with torch.no_grad():
            output = model(rgb, lr)
            loss_psnr = criterion_PSNR(output, hr)
            loss_ssim = criterion_SSIM(output, hr)
            losses_psnr.update(loss_psnr.data)
            losses_ssim.update(loss_ssim.data)
    return losses_psnr.avg, losses_ssim.avg


def main():
    config = load_config()
    setup_environment(config)
    writer = initialize_tensorboard(config['outf'])
    
    print("\nLoading dataset...")
    train_loader, val_loader = load_datasets(config)

    criterion_L1 = nn.L1Loss().cuda()
    criterion_PSNR = Loss_PSNR().cuda()
    criterion_SSIM = SSIM().cuda()
    criterion_Perceptual = VGGPerceptualLoss().cuda()

    model = initialize_model()

    optimizer = optim.Adam(model.parameters(), lr=float(config['init_lr']), betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000 * config['end_epoch'], eta_min=1e-6)

    start_epoch, iteration = load_checkpoint(model, optimizer, config['resume_file'])
    total_iteration = 1000 * config['end_epoch']
    best_psnr, best_ssim = 0, 0
    prev_time = time.time()

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, delta=0.001)

    train_losses = []
    val_losses = []

    while iteration < total_iteration:
        model.train()
        losses = AverageMeter()
        losses_l1 = AverageMeter()
        losses_ssim = AverageMeter()
        losses_perceptual = AverageMeter()

        for lr, rgb, hr in train_loader:
            lr, rgb, hr = lr.cuda(), rgb.cuda(), hr.cuda()
            optimizer.zero_grad()
            output = model(rgb, lr)
            l1_loss = criterion_L1(output, hr)
            ssim_loss = 1 - criterion_SSIM(output, hr)
            perceptual_loss = criterion_Perceptual(output, hr)
            loss = 7 * l1_loss + ssim_loss + 0.15 * perceptual_loss

            loss.backward()
            optimizer.step()
            scheduler.step()

            losses.update(loss.data)
            losses_l1.update(l1_loss.data)
            losses_ssim.update(ssim_loss.data)
            losses_perceptual.update(perceptual_loss.data)
            iteration += 1

            if iteration % 100 == 0:
                print(f'[Iter: {iteration}/{total_iteration}], LR={optimizer.param_groups[0]["lr"]:.9f}, '
                      f'Train Loss={losses.avg:.9f}, L1 Loss={losses_l1.avg:.9f}, '
                      f'SSIM Loss={losses_ssim.avg:.9f}, Perceptual Loss={losses_perceptual.avg:.9f}')

            if iteration % (1000 * (16 // config['batch_size'])) == 0:
                psnr, ssim = validate(model, val_loader, criterion_PSNR, criterion_SSIM)
                print(f'[Epoch: {iteration // 1000}/{config["end_epoch"]}], PSNR={psnr:.4f}, SSIM={ssim:.4f}')

                writer.add_scalar('Train Loss', losses.avg, iteration // 1000)
                writer.add_scalar('L1 Loss', losses_l1.avg, iteration // 1000)
                writer.add_scalar('SSIM Loss', losses_ssim.avg, iteration // 1000)
                writer.add_scalar('Perceptual Loss', losses_perceptual.avg, iteration // 1000)
                writer.add_scalar('Learning Rate', optimizer.param_groups[0]['lr'], iteration // 1000)
                writer.add_scalar('PSNR', psnr, iteration // 1000)
                writer.add_scalar('SSIM', ssim, iteration // 1000)

                if (psnr > best_psnr or ssim > best_ssim) and iteration // 1000 > 15:
                    best_psnr = max(psnr, best_psnr)
                    best_ssim = max(ssim, best_ssim)
                    save_checkpoint(config['outf'], iteration // 1000, iteration, model, optimizer)

                # Early stopping check
                if early_stopping.step(val_loss=ssim):
                    print("Training stopped early.")
                    break

            # Log images periodically
            if iteration % 1000 == 0:
                log_images(writer, iteration, lr, rgb, output, hr)

            # Record losses for plotting
            train_losses.append(losses.avg)
            val_losses.append(ssim)

            iters_done = iteration
            iters_left = total_iteration - iters_done
            time_left = datetime.timedelta(seconds=iters_left * (time.time() - prev_time))
            prev_time = time.time()
            print(f'Time left: {time_left}')

        # Plot the loss curves
        if iteration % 1000 == 0:
            plot_loss(train_losses, val_losses, iteration)

    print("Training complete.")


if __name__ == '__main__':
    main()


import torch
import torch.nn as nn
import os
import torchvision


class AverageMeter:
    """Computes and stores the average and current value."""  
    def __init__(self):
        self.reset()

    def reset(self):
        """Resets all statistics."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """Updates the statistics with new values."""
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class L1Loss(nn.Module):
    """L1 Loss implementation."""  
    def __init__(self):
        super(L1Loss, self).__init__()
        self.loss = nn.L1Loss()

    def forward(self, x, y):
        """Forward pass for L1 loss."""
        return self.loss(x, y)


class Loss_PSNR(nn.Module):
    """PSNR Loss implementation."""  
    def __init__(self):
        super(Loss_PSNR, self).__init__()

    def forward(self, im_true, im_fake, data_range=255):
        """Forward pass for PSNR loss."""
        Itrue = im_true.clamp(0., 1.).mul_(data_range)
        Ifake = im_fake.clamp(0., 1.).mul_(data_range)
        mse = torch.mean((Itrue - Ifake) ** 2) + 1e-6  # Small epsilon for stability
        return 20 * torch.log10(data_range / torch.sqrt(mse))


def save_checkpoint(model_path, epoch, iteration, model, optimizer):
    """Saves the model checkpoint."""
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    
    state = {
        'epoch': epoch,
        'iter': iteration,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(state, os.path.join(model_path, f'net_{epoch}epoch.pth'))


class VGGPerceptualLoss(nn.Module):
    """VGG Perceptual Loss implementation."""  
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.blocks = nn.ModuleList([
            torchvision.models.vgg16(pretrained=True).features[:4].eval(),
            torchvision.models.vgg16(pretrained=True).features[4:9].eval(),
            torchvision.models.vgg16(pretrained=True).features[9:16].eval(),
            torchvision.models.vgg16(pretrained=True).features[16:23].eval(),
        ])
        
        for block in self.blocks:
            block = block.to(self.device)
            for p in block.parameters():
                p.requires_grad = False
                
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device))

    def forward(self, input, target, feature_layers=[0, 1, 2, 3], style_layers=[]):
        """Forward pass for VGG perceptual loss."""
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
            
        input = (input - self.mean) / self.std
        target = (target - self.mean) / self.std
        
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
            
        loss = 0.0
        x = input
        y = target
        
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            
            if i in feature_layers:
                loss += nn.functional.l1_loss(x, y)
                
            if i in style_layers:
                act_x = x.view(x.size(0), x.size(1), -1)
                act_y = y.view(y.size(0), y.size(1), -1)
                gram_x = act_x.bmm(act_x.transpose(1, 2))
                gram_y = act_y.bmm(act_y.transpose(1, 2))
                loss += nn.functional.l1_loss(gram_x, gram_y)
                
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class FlexibleWindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super(FlexibleWindowAttention, self).__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.softmax = nn.Softmax(dim=-1)
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = self.softmax(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

class ChannelSpatialAttentionBlock(nn.Module):
    def __init__(self, dim):
        super(ChannelSpatialAttentionBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.activation = nn.SiLU()
        self.channel_att = nn.AdaptiveAvgPool2d(1)
        self.spatial_att = nn.Conv2d(2, 1, kernel_size=7, padding=3)

    def forward(self, x):
        b, c, h, w = x.size()
        avg_pool = self.channel_att(x).view(b, c, 1, 1)
        max_pool = F.adaptive_max_pool2d(x, 1).view(b, c, 1, 1)
        channel_att = torch.sigmoid(self.conv1(avg_pool) + self.conv2(max_pool))
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.sigmoid(self.spatial_att(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att
        return x

class FWSAT(nn.Module):
    def __init__(self, dim, num_heads, window_size, num_layers):
        super(FWSAT, self).__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                FlexibleWindowAttention(dim, window_size, num_heads),
                ChannelSpatialAttentionBlock(dim)
            ) for _ in range(num_layers)
        ])
        self.conv_in = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(dim, 3, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv_in(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(1, 2)
        
        for layer in self.layers:
            x = layer(x) + x
            
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.conv_out(x)
        return x

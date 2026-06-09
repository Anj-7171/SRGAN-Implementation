import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return x + self.block(x)   # skip connection


class UpsampleBlock(nn.Module):
    def __init__(self, in_c, scale=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, in_c * scale**2, 3, 1, 1),
            nn.PixelShuffle(scale),   # sub-pixel convolution
            nn.PReLU(),
        )

    def forward(self, x):
        return self.block(x)


class Generator(nn.Module):
    """SRGAN Generator — 4x upscaling by default (two 2x upsample blocks)."""
    def __init__(self, in_channels=3, num_channels=64, num_residuals=16):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, num_channels, 9, 1, 4),
            nn.PReLU(),
        )
        self.residuals = nn.Sequential(*[ResidualBlock(num_channels) for _ in range(num_residuals)])
        self.post_res = nn.Sequential(
            nn.Conv2d(num_channels, num_channels, 3, 1, 1),
            nn.BatchNorm2d(num_channels),
        )
        self.upsample = nn.Sequential(
            UpsampleBlock(num_channels, scale=2),
            UpsampleBlock(num_channels, scale=2),
        )
        self.final = nn.Conv2d(num_channels, in_channels, 9, 1, 4)

    def forward(self, x):
        initial = self.initial(x)
        x = self.residuals(initial)
        x = self.post_res(x) + initial   # global skip connection
        x = self.upsample(x)
        return torch.tanh(self.final(x))
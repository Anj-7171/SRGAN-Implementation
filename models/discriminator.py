import torch.nn as nn

def conv_block(in_c, out_c, stride):
    return nn.Sequential(
        nn.Conv2d(in_c, out_c, 3, stride, 1),
        nn.BatchNorm2d(out_c),
        nn.LeakyReLU(0.2),
    )

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, features=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels, features, 3, 1, 1),
            nn.LeakyReLU(0.2),
            conv_block(features,       features,   2),
            conv_block(features,       features*2, 1),
            conv_block(features*2,     features*2, 2),
            conv_block(features*2,     features*4, 1),
            conv_block(features*4,     features*4, 2),
            conv_block(features*4,     features*8, 1),
            conv_block(features*8,     features*8, 2),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
            nn.Linear(features * 8 * 6 * 6, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 1),
        )

    def forward(self, x):
        return self.net(x)
import torch.nn as nn
from torchvision.models import vgg19, VGG19_Weights

class VGGFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT)
        # Use features up to relu5_4 (layer index 36)
        self.features = nn.Sequential(*list(vgg.features)[:36]).eval()
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.features(x)
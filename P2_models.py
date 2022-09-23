from turtle import forward

import torch
import torch.nn as nn
from torchvision.models import VGG, VGG16_Weights, vgg16


class FCN32s(nn.Module):
    def __init__(self, extractor: VGG, n_class=7) -> None:
        super().__init__()
        self.features = extractor.features
        # replace mlp by conv
        self.pixel_clf = nn.Sequential(
            nn.Conv2d(512, 4096, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4096, 4096, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(4096, n_class, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.ConvTranspose2d(
                n_class, n_class, kernel_size=64, stride=30, padding=1),
        )

    def forward(self, x) -> torch.Tensor:
        x = self.features(x)
        x = self.pixel_clf(x)
        return x


if __name__ == '__main__':
    net = FCN32s(vgg16(weights=VGG16_Weights.DEFAULT)).cuda()
    net(torch.rand(16, 3, 512, 512).cuda())

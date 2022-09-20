import torch
from torch import nn


class ALL_Conv(nn.Module):
    def __init__(self, in_channel=3, out_size=50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv2d(in_channel, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            # replaces pooling layer
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(96, 96, 3, padding=1),
            nn.ReLU(),
            # replaces pooling layer
            nn.Conv2d(96, 96, 3, padding=1, stride=2),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Conv2d(96, 192, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Conv2d(192, out_size, 1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):
        out = self.net(x)  # shape = (b, 10, 1, 1)
        out = self.head(out)
        out = torch.squeeze(out)
        return out  # shape = (b, 10)

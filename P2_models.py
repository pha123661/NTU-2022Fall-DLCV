import torch
import torch.nn as nn
from torchvision.models import VGG16_Weights, vgg16


class my_FCN32s(nn.Module):
    def __init__(self, n_class=7) -> None:
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features[0].padding = 100

        # replace fc by conv
        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 2048, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(2048, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.score_fr = nn.Conv2d(4096, n_class, 1)
        self.upscore = nn.ConvTranspose2d(
            n_class, n_class, 64, stride=32)

    def forward(self, x) -> torch.Tensor:
        raw_h, raw_w = x.shape[2], x.shape[3]
        x = self.features(x)
        x = self.fc6(x)
        x = self.fc7(x)
        x = self.score_fr(x)
        x = self.upscore(x)
        # crop to original size
        x = x[..., x.shape[2] - raw_h:, x.shape[3] - raw_w:]
        return x


def FCN32s():
    return my_FCN32s()


if __name__ == '__main__':
    net = FCN32s().cuda()
    print(net(torch.rand(8, 3, 512, 512).cuda()).shape)

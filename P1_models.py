import math

import torch
from torch import nn

# class ALL_Conv(nn.Module):
#     def __init__(self, in_channel=3, out_size=50):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Dropout(0.2),
#             nn.Conv2d(in_channel, 96, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(96, 96, 3, padding=1),
#             nn.ReLU(),
#             # replaces pooling layer
#             nn.Conv2d(96, 96, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Dropout(0.5),

#             nn.Conv2d(96, 96, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(96, 96, 3, padding=1),
#             nn.ReLU(),
#             # replaces pooling layer
#             nn.Conv2d(96, 96, 3, padding=1, stride=2),
#             nn.ReLU(),
#             nn.Dropout(0.5),

#             nn.Conv2d(96, 192, 3, padding=1),
#             nn.ReLU(),
#             nn.Conv2d(192, 192, 1),
#             nn.ReLU(),
#         )
#         self.head = nn.Sequential(
#             nn.Conv2d(192, out_size, 1),
#             nn.ReLU(),
#             nn.AdaptiveAvgPool2d(1),
#         )

#     def forward(self, x):
#         out = self.net(x)  # shape = (b, 10, 1, 1)
#         out = self.head(out)
#         out = torch.squeeze(out)
#         return out  # shape = (b, 10)


class VGG(nn.Module):
    '''
    VGG model
    '''

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 512),
        )
        self.classifier2 = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(512, 50),
        )
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = self.classifier2(x)
        return x

    def get_embedding(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    '''returns layer structure of VGG with different confiurations'''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


vgg_cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M',
          512, 512, 512, 512, 'M'],
}


def vgg11():
    return VGG(make_layers(vgg_cfg['A']))


def vgg11_bn():
    return VGG(make_layers(vgg_cfg['A'], batch_norm=True))


def vgg13():
    return VGG(make_layers(vgg_cfg['B']))


def vgg13_bn():
    return VGG(make_layers(vgg_cfg['B'], batch_norm=True))


def vgg16():
    return VGG(make_layers(vgg_cfg['D']))


def vgg16_bn():
    return VGG(make_layers(vgg_cfg['D'], batch_norm=True))


def vgg19():
    return VGG(make_layers(vgg_cfg['E']))


def vgg19_bn():
    return VGG(make_layers(vgg_cfg['E'], batch_norm=True))

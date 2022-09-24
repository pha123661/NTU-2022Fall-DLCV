import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights, vgg16


########## P2B ##########
class U_block(nn.Module):
    '''building block of each stage'''

    def __init__(self, in_chan, out_chan) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 3),
        )

    def forward(self, x):
        return self.net(x)


class U_Encoder(nn.Module):
    '''Encoder of U-Net'''

    def __init__(self, chans=[3, 64, 128, 256, 512, 1024]) -> None:
        super().__init__()
        self.chans = chans
        self.blocks = nn.ModuleList()
        for idx in range(len(chans) - 1):
            self.blocks.append(U_block(chans[idx], chans[idx + 1]))

    def forward(self, x):
        ret = []
        for block in self.blocks:
            x = block(x)
            ret.append(x)
            x = nn.functional.max_pool2d(x, kernel_size=2)
        return ret


class U_Decoder(nn.Module):
    '''Decoder of U-Net'''

    def __init__(self, chans=[1024, 512, 256, 128, 64]) -> None:
        super().__init__()
        self.chans = chans
        self.blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for idx in range(len(chans) - 1):
            self.blocks.append(U_block(chans[idx], chans[idx + 1]))
            self.up_convs.append(nn.ConvTranspose2d(
                chans[idx], chans[idx + 1], 2, 2))

    def forward(self, skip_cons, x):
        def crop_feature(feature, shape):
            _, _, H, W = shape
            return torchvision.transforms.CenterCrop([H, W])(feature)

        for block, up_conv, feature in zip(self.blocks, self.up_convs, skip_cons):
            x = up_conv(x)
            '''
            input shape = (b, 3, 512, 512)
            feature.shape                  x.shape
            torch.Size([1, 512, 56, 56])   torch.Size([1, 512, 48, 48])
            torch.Size([1, 256, 121, 121]) torch.Size([1, 256, 88, 88])
            torch.Size([1, 128, 250, 250]) torch.Size([1, 128, 168, 168])
            torch.Size([1, 64, 508, 508])  torch.Size([1, 64, 328, 328])
            '''
            feature = crop_feature(feature, x.shape)
            x = torch.cat((feature, x), dim=1)
            x = block(x)

        return x


class U_Net(nn.Module):
    def __init__(
        self,
        enc_chans=[3, 64, 128, 256, 512, 1024],
        dec_chans=[1024, 512, 256, 128, 64],
        n_class=7,
    ) -> None:
        super().__init__()
        self.Encoder, self.Decoder = U_Encoder(enc_chans), U_Decoder(dec_chans)
        self.clf = nn.Conv2d(dec_chans[-1], n_class, 1)

    def forward(self, x):
        img_size = tuple(x.shape[-2:])

        features = self.Encoder(x)
        features.reverse()  # from down to top
        # bottom feature map doesn't need concatenation
        x = self.Decoder(features[1:], features[0])
        x = self.clf(x)
        x = nn.functional.interpolate(x, img_size)
        return x


if __name__ == '__main__':
    net = U_Net().cuda()
    ret = net(torch.rand(1, 3, 512, 512).cuda())
    print(ret.shape)


########## P2A ##########


class my_FCN32s(nn.Module):
    def __init__(self, n_class=7) -> None:
        super().__init__()
        self.features = vgg16(weights=VGG16_Weights.DEFAULT).features
        self.features[0].padding = 100

        # replace fc by conv
        # fc6
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.fc6[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[0].weight.data.view(
            self.fc6[0].weight.size())
        self.fc6[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[0].bias.data.view(self.fc6[0].bias.size())

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
        )

        self.fc7[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[3].weight.data.view(
            self.fc7[0].weight.size())
        self.fc7[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[3].bias.data.view(self.fc7[0].bias.size())

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

import timm
import torch
import torch.nn as nn
import torchvision
from torchvision.models import VGG16_Weights, vgg16
from torchvision.models.segmentation.deeplabv3 import DeepLabHead, FCNHead


########## P2B ##########
class U_block(nn.Module):
    '''building block of each stage'''

    def __init__(self, in_chan, out_chan) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 3),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
            nn.Conv2d(out_chan, out_chan, 3),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True),
        )

    def forward(self, x):
        return self.net(x)


class U_Decoder(nn.Module):
    '''Decoder of U-Net'''

    def __init__(self, chans) -> None:
        super().__init__()
        self.chans = chans
        self.blocks = nn.ModuleList()
        self.up_convs = nn.ModuleList()
        for idx in range(len(chans) - 1):
            self.up_convs.append(nn.ConvTranspose2d(
                chans[idx], chans[idx + 1], 2, 2))
            self.blocks.append(U_block(2 * chans[idx + 1], chans[idx + 1]))

    def forward(self, skip_cons, x):
        def crop_feature(feature, shape):
            _, _, H, W = shape
            return torchvision.transforms.CenterCrop([H, W])(feature)

        for block, up_conv, feature in zip(self.blocks, self.up_convs, skip_cons):
            # print("#########################")
            # print("x:", x.shape)
            # x = up_conv(x)
            # print("x_up:", x.shape)
            # feature = crop_feature(feature, x.shape)
            # print("f:", feature.shape)
            # x = torch.cat((feature, x), dim=1)
            # print('cat:', x.shape)
            # print(block)
            # x = block(x)

            x = up_conv(x)
            feature = crop_feature(feature, x.shape)
            x = torch.cat((feature, x), dim=1)
            x = block(x)

        return x


class U_Net(nn.Module):
    def __init__(
        self,
        n_class=7,
    ) -> None:
        super().__init__()
        self.Encoder = timm.create_model(
            'resnet34', features_only=True, pretrained=True)
        dec_chans = self.Encoder.feature_info.channels()
        print("U_net chans:", dec_chans)
        dec_chans.reverse()
        self.center_block = nn.Sequential(
            nn.Conv2d(dec_chans[0], dec_chans[0], 3, padding=1),
            nn.BatchNorm2d(dec_chans[0]),
            nn.ReLU(True)
        )
        self.Decoder = U_Decoder(dec_chans)
        self.clf = nn.Conv2d(dec_chans[-1], n_class, 1)

        print("U_net #param:", sum(p.numel() for p in self.parameters()))

    def forward(self, x):
        def crop_feature(feature, shape):
            _, _, H, W = shape
            return torchvision.transforms.CenterCrop([H, W])(feature)
        in_shape = x.shape[:]

        features = self.Encoder(x)
        features.reverse()  # from down to top
        skips = features[1:]
        x = features[0]
        x = self.center_block(x)
        # bottom feature map doesn't need concatenation
        x = self.Decoder(skips, x)
        x = self.clf(x)
        x = crop_feature(x, in_shape)
        return x


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
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc6[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[0].weight.data.view(
            self.fc6[0].weight.size())
        self.fc6[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[0].bias.data.view(self.fc6[0].bias.size())

        # fc7
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.ReLU(),
            nn.Dropout2d(),
        )

        self.fc7[0].weight.data = vgg16(weights=VGG16_Weights.DEFAULT).classifier[3].weight.data.view(
            self.fc7[0].weight.size())
        self.fc7[0].bias.data = vgg16(
            weights=VGG16_Weights.DEFAULT).classifier[3].bias.data.view(self.fc7[0].bias.size())

        self.score_fr = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
            nn.ReLU()
        )

        self.upscore = nn.Sequential(
            nn.ConvTranspose2d(n_class, n_class, 64, stride=32)
        )

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


def DeepLabv3(n_classes=7, mode='resnet'):
    if mode == 'resnet':
        model = torch.hub.load('pytorch/vision:v0.10.0',
                               'deeplabv3_resnet50', pretrained=True)
        model.classifier = DeepLabHead(2048, n_classes)
        model.aux_classifier = FCNHead(1024, n_classes)
    elif mode == 'mobile':
        model = torch.hub.load('pytorch/vision:v0.10.0',
                               'deeplabv3_mobilenet_v3_large', pretrained=True)
        model.classifier = DeepLabHead(960, n_classes)
        model.aux_classifier = FCNHead(40, n_classes)

    model.train()
    return model


if __name__ == '__main__':
    model = DeepLabv3(7)
    model.train()
    print(model(torch.rand(3, 3, 512, 512))['aux'].shape)

    # net = FCN32s().cuda()
    # ret = net(torch.rand(1, 3, 512, 512).cuda())
    # pytorch_total_params = sum(p.numel() for p in net.parameters())
    # print(pytorch_total_params)
    # print(ret.shape)

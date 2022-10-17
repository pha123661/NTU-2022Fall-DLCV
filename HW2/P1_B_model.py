import torch
import torch.nn as nn


def DCGAN_init(layer):
    if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d)):
        nn.init.normal_(layer.weight.data, 0, 0.02)
    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, (nn.LeakyReLU, nn.Tanh, nn.Sigmoid, nn.Sequential, DCGAN_G, SNGAN_D)):
        pass
    else:
        raise ModuleNotFoundError(
            f"initialize G error: {type(layer)}: {layer}")


class DCGAN_G(nn.Module):
    def __init__(self, latent_size=100, num_map=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_size, num_map * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(num_map * 8),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(num_map * 8, num_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map * 4),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(num_map * 4, num_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map * 2),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(num_map * 2, num_map, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose2d(num_map, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.apply(DCGAN_init)

    def forward(self, z):
        return self.net(z)


class SNGAN_D(nn.Module):
    def __init__(self, num_map=64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, num_map, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_map, num_map * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map * 2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_map * 2, num_map * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map * 4),
            nn.LeakyReLU(0.2),

            nn.Conv2d(num_map * 4, num_map * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(num_map * 8),
            nn.LeakyReLU(0.2),

            # global conv
            nn.Conv2d(num_map * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

        self.apply(DCGAN_init)
        add_SN(self)

    def forward(self, img):
        return self.net(img)


def add_SN(net):
    for name, layer in net.named_children():
        net.add_module(name, add_SN(layer))
    if isinstance(net, (nn.Conv2d, nn.Linear)):
        return nn.utils.spectral_norm(net)
    else:
        return net


if __name__ == "__main__":
    net = DCGAN_G()
    print(net)
    print(net(torch.rand(16, 100, 1, 1)).shape)

    net = SNGAN_D()
    print(net)
    print(net(torch.rand(16, 3, 64, 64)).shape)

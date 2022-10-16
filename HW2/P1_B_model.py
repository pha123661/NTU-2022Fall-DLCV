import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, latent_size=128, out_chans=3, n_featuremap=64, n_upsamplings=4) -> None:
        super().__init__()

        def UpsampleLayer(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, kernel_size, stride=stride,
                                   padding=padding, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        Layers = []

        # 1: 1x1 -> 4x4
        d = min(n_featuremap * 2 ** (n_upsamplings - 1), n_featuremap * 8)
        Layers.append(
            UpsampleLayer(latent_size, d, kernel_size=4, stride=1, padding=0)
        )

        # 2: upsamplings, 4x4 -> 8x8 -> 16x16 -> ...
        for i in range(n_upsamplings - 1):
            d_last = d
            d = min(n_featuremap * 2 **
                    (n_upsamplings - 2 - i), n_featuremap * 8)
            Layers.append(
                UpsampleLayer(d_last, d, kernel_size=4, stride=2, padding=1)
            )

        Layers.append(
            nn.ConvTranspose2d(d, out_chans, kernel_size=4,
                               stride=2, padding=1)
        )
        Layers.append(nn.Tanh())

        self.gen = nn.Sequential(*Layers)

    def forward(self, z):
        x = self.gen(z)
        return x


class Discriminator(nn.Module):
    def __init__(self, in_chans=3, n_featuremap=64, n_downsamplings=4) -> None:
        super().__init__()

        def DownsampleLayer(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride,
                          padding=padding, bias=False),
                # gradient penalty is not compatible with batch norm
                nn.GroupNorm(1, out_dim),  # layer norm
                nn.LeakyReLU(0.2)
            )

        layers = []

        # 1: downsamplings, ... -> 16x16 -> 8x8 -> 4x4
        d = n_featuremap
        layers.append(nn.Conv2d(in_chans, d,
                      kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2))

        for i in range(n_downsamplings - 1):
            d_last = d
            d = min(n_featuremap * 2 ** (i + 1), n_featuremap * 8)
            layers.append(DownsampleLayer(
                d_last, d, kernel_size=4, stride=2, padding=1))

        # 2: logit
        layers.append(nn.Conv2d(d, 1, kernel_size=4, stride=1, padding=0))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = self.net(x)
        return y

    def calc_gp(self, real, fake):
        def get_sample(real, fake):
            # # DRAGAN sampling, fake is less usefull
            # beta = torch.rand_like(real)
            # fake = real + 0.5 * real.std() * beta

            shape = [real.size(0)] + [1] * (real.dim() - 1)
            alpha = torch.rand(shape, device=real.device)
            sample = real + alpha * (fake - real)
            return sample

        def penalty(grad):
            grad_norm = grad.reshape(grad.size(0), -1).norm(p=2, dim=1)
            gp = ((grad_norm - 1)**2).mean()
            return gp

        x = get_sample(real, fake).detach()
        x.requires_grad = True
        logits = self(x)
        grad = torch.autograd.grad(
            logits, x, grad_outputs=torch.ones_like(logits), create_graph=True)[0]
        gp = penalty(grad)
        return gp


if __name__ == '__main__':
    print(Generator())
    print(Discriminator())

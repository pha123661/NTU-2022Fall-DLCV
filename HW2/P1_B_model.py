import torch
from torch import nn


def DCGAN_init(layer):
    if isinstance(layer, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
        nn.init.normal_(layer.weight.data, 0, 0.02)
    elif isinstance(layer, (nn.BatchNorm2d, nn.BatchNorm1d, nn.InstanceNorm2d, )):
        nn.init.normal_(layer.weight.data, 1.0, 0.02)
        nn.init.constant_(layer.bias.data, 0)
    elif isinstance(layer, (nn.LeakyReLU, nn.ReLU, nn.Tanh, nn.Sigmoid, nn.Sequential, Generator, Discriminator)):
        pass
    else:
        raise ModuleNotFoundError(
            f"initialize Module error: {type(layer)}: {layer}")


class Generator(nn.Module):
    # def __init__(self, latent_size=128, num_map=64) -> None:
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.ConvTranspose2d(latent_size, num_map * 8, 4, 1, 0, bias=False),
    #         nn.BatchNorm2d(num_map * 8),
    #         nn.ReLU(True),

    #         nn.ConvTranspose2d(num_map * 8, num_map * 4, 4, 2, 1, bias=False),
    #         nn.BatchNorm2d(num_map * 4),
    #         nn.ReLU(True),

    #         nn.ConvTranspose2d(num_map * 4, num_map * 2, 4, 2, 1, bias=False),
    #         nn.BatchNorm2d(num_map * 2),
    #         nn.ReLU(True),

    #         nn.ConvTranspose2d(num_map * 2, num_map, 4, 2, 1, bias=False),
    #         nn.BatchNorm2d(num_map),
    #         nn.ReLU(True),

    #         nn.ConvTranspose2d(num_map, 3, 4, 2, 1, bias=False),
    #         nn.Tanh()
    #     )
    #     self.apply(DCGAN_init)

    # def forward(self, z):
    #     return self.net(z)
    def __init__(self, latent_size=128, n_featuremap=64) -> None:
        super().__init__()

        def UpsampleLayer(in_dim, out_dim):
            return nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 5, 2,
                                   padding=2, output_padding=1, bias=False),
                nn.BatchNorm2d(out_dim),
                nn.ReLU()
            )

        self.l1 = nn.Sequential(
            nn.Linear(latent_size, n_featuremap * 8 * 4 * 4, bias=False),
            nn.BatchNorm1d(n_featuremap * 8 * 4 * 4),
            nn.ReLU()
        )
        self.l2_5 = nn.Sequential(
            UpsampleLayer(n_featuremap * 8, n_featuremap * 4),
            UpsampleLayer(n_featuremap * 4, n_featuremap * 2),
            UpsampleLayer(n_featuremap * 2, n_featuremap),
            nn.ConvTranspose2d(n_featuremap, 3, 5, 2,
                               padding=2, output_padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        y = self.l1(x)
        y = y.view(y.size(0), -1, 4, 4)
        y = self.l2_5(y)
        return y


class Discriminator(nn.Module):
    # def __init__(self, num_map=64) -> None:
    #     super().__init__()
    #     self.net = nn.Sequential(
    #         nn.Conv2d(3, num_map, 4, 2, 1, bias=False),

    #         nn.Conv2d(num_map, num_map * 2, 4, 2, 1, bias=False),
    #         nn.InstanceNorm2d(num_map * 2, affine=True),
    #         nn.LeakyReLU(0.2),

    #         nn.Conv2d(num_map * 2, num_map * 4, 4, 2, 1, bias=False),
    #         nn.InstanceNorm2d(num_map * 4, affine=True),
    #         nn.LeakyReLU(0.2),

    #         nn.Conv2d(num_map * 4, num_map * 8, 4, 2, 1, bias=False),
    #         nn.InstanceNorm2d(num_map * 8, affine=True),
    #         nn.LeakyReLU(0.2),

    #         # global conv
    #         nn.Conv2d(num_map * 8, 1, 4, 1, 0, bias=False),
    #     )

    #     self.apply(DCGAN_init)

    # def forward(self, img):
    #     return self.net(img)

    def __init__(self, in_chans=3, n_featuremap=64) -> None:
        super().__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2)
            )

        self.ls = nn.Sequential(
            nn.Conv2d(in_chans, n_featuremap, 5, 2, 2), nn.LeakyReLU(0.2),
            conv_ln_lrelu(n_featuremap, n_featuremap * 2),
            conv_ln_lrelu(n_featuremap * 2, n_featuremap * 4),
            conv_ln_lrelu(n_featuremap * 4, n_featuremap * 8),
            nn.Conv2d(n_featuremap * 8, 1, 4))

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
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
    print(g := Generator())
    g.apply(DCGAN_init)
    print(d := Discriminator())
    d.apply(DCGAN_init)
    print(d(torch.rand(64, 3, 64, 64)).shape)

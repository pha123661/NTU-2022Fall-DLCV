import numpy as np
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_chans, out_chans, residual=False) -> None:
        super().__init__()
        self.residual = residual
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_chans, out_chans, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_chans, out_chans,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_chans),
            nn.GELU()
        )

    def forward(self, x):
        if self.residual:
            x1 = self.conv1(x)
            x2 = self.conv2(x1)
            if x.shape[1] == x2.shape[1]:
                out = x + x2
            else:
                out = x1 + x2
            return out / 1.414
        else:
            x = self.conv1(x)
            x = self.conv2(x)
            return x


class U_encoder(nn.Module):
    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(in_chans, out_chans),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.net(x)


class U_decoder(nn.Module):
    def __init__(self, in_chans, out_chans) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(in_chans, out_chans, kernel_size=2, stride=2),
            ConvBlock(out_chans, out_chans),
            ConvBlock(out_chans, out_chans)
        )

    def forward(self, x, skip):
        x = torch.cat((x, skip), 1)
        x = self.net(x)
        return x


class EmbedFC(nn.Module):
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x):
        x = x.reshape(-1, self.in_dim)
        return self.net(x)


class Unet(nn.Module):
    def __init__(self, in_chans, n_features, n_classes) -> None:
        super().__init__()
        self.in_chans = in_chans
        self.n_features = n_features
        self.n_classes = n_classes

        self.init_conv = ConvBlock(in_chans, n_features, residual=True)

        self.encode1 = U_encoder(n_features, n_features)
        self.encode2 = U_encoder(n_features, 2 * n_features)

        self.to_vec = nn.Sequential(
            nn.AvgPool2d(7),
            nn.GELU()
        )

        self.time_embed1 = EmbedFC(1, 2 * n_features)
        self.time_embed2 = EmbedFC(1, n_features)

        self.contextembed1 = EmbedFC(n_classes, 2 * n_features)
        self.contextembed2 = EmbedFC(n_classes, n_features)

        self.decode0 = nn.Sequential(
            nn.ConvTranspose2d(2 * n_features, 2 * n_features,
                               kernel_size=7, stride=7),
            nn.GroupNorm(8, 2 * n_features),
            nn.ReLU(True),
        )
        self.decode1 = U_decoder(4 * n_features, n_features)
        self.decode2 = U_decoder(2 * n_features, n_features)
        self.out = nn.Sequential(
            nn.Conv2d(2 * n_features, n_features,
                      kernel_size=3, stride=1, padding=1),
            nn.GroupNorm(8, n_features),
            nn.ReLU(True),
            nn.Conv2d(n_features, self.in_chans,
                      kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, c, t, context_mask):
        x = self.init_conv(x)
        enc1 = self.encode1(x)
        enc2 = self.encode2(enc1)
        hidden_vec = self.to_vec(enc2)

        # convert context
        c = nn.functional.one_hot(
            c, num_classes=self.n_classes).type(torch.float)
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1, self.n_classes)
        context_mask = -(1 - context_mask)  # flip 01
        # context_mask is for context dropout
        c *= context_mask

        # embed context and time step
        c_emb1 = self.contextembed1(c).reshape(-1, self.n_features * 2, 1, 1)
        t_emb1 = self.time_embed1(t).reshape(-1, self.n_features * 2, 1, 1)
        c_emb2 = self.contextembed2(c).reshape(-1, self.n_features, 1, 1)
        t_emb2 = self.time_embed2(t).reshape(-1, self.n_features, 1, 1)

        dec1 = self.decode0(hidden_vec)
        dec2 = self.decode1(x=c_emb1 * dec1 + t_emb1, skip=enc2)
        dec3 = self.decode2(x=c_emb2 * dec2 + t_emb2, skip=enc1)
        out = self.out(torch.cat((dec3, x), 1))

        return out


def ddpm_schedules(beta1, beta2, T):
    assert beta1 < beta2 < 1.0

    beta_t = (beta2 - beta1) * torch.arange(0, T +
                                            1, dtype=torch.float32) / T + beta1
    sqrt_beta_t = torch.sqrt(beta_t)
    alpha_t = 1 - beta_t
    log_alpha_t = torch.log(alpha_t)
    alphabar_t = torch.cumsum(log_alpha_t, dim=0).exp()

    sqrtab = torch.sqrt(alphabar_t)
    oneover_sqrta = 1 / torch.sqrt(alpha_t)

    sqrtmab = torch.sqrt(1 - alphabar_t)
    mab_over_sqrtmab_inv = (1 - alpha_t) / sqrtmab

    return {
        "alpha_t": alpha_t,  # \alpha_t
        "oneover_sqrta": oneover_sqrta,  # 1/\sqrt{\alpha_t}
        "sqrt_beta_t": sqrt_beta_t,  # \sqrt{\beta_t}
        "alphabar_t": alphabar_t,  # \bar{\alpha_t}
        "sqrtab": sqrtab,  # \sqrt{\bar{\alpha_t}}
        "sqrtmab": sqrtmab,  # \sqrt{1-\bar{\alpha_t}}
        # (1-\alpha_t)/\sqrt{1-\bar{\alpha_t}}
        "mab_over_sqrtmab": mab_over_sqrtmab_inv,
    }


class DDPM_framework(nn.Module):
    def __init__(self, network, betas, n_T, device, drop_prob=0.1) -> None:
        super().__init__()
        self.n_T = n_T
        self.device = device
        self.drop_prob = drop_prob
        self.loss_fn = nn.MSELoss()

        self.net = network.to(self.device)

        for k, v in ddpm_schedules(betas[0], betas[1], self.n_T).items():
            # buffers are saved with model, but not updating it
            self.register_buffer(k, v)

    def forward(self, x, c):
        _ts = torch.randint(1, self.n_T, (x.shape[0], )).to(self.device)
        noise = torch.randn(*x.shape, device=self.device)

        x_t = self.sqrtab[_ts, None, None, None] * x + \
            self.sqrtmab[_ts, None, None, None] * noise

        # context dropout
        context_mask = torch.bernoulli(torch.ones(
            *c.shape) * self.drop_prob).to(self.device)

        return self.loss_fn(noise, self.net(x_t, c, _ts / self.n_T, context_mask))

    def class_gen(self, n_sample, size, device, class_idx, guide_w=0.5):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.ones(n_sample, device=device,
                         dtype=torch.int64) * class_idx

        context_mask = torch.zeros(*c_i.shape).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1

        # history
        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.net(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) +
                self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

    def sample(self, n_sample, size, device, guide_w=0.0):
        x_i = torch.randn(n_sample, *size).to(device)
        c_i = torch.arange(0, 10).to(device)
        c_i = c_i.repeat(n_sample // c_i.shape[0])

        context_mask = torch.zeros(*c_i.shape).to(device)

        c_i = c_i.repeat(2)
        context_mask = context_mask.repeat(2)
        context_mask[n_sample:] = 1

        # history
        x_i_store = []
        for i in range(self.n_T, 0, -1):
            t_is = torch.tensor([i / self.n_T]).to(device)
            t_is = t_is.repeat(n_sample, 1, 1, 1)

            # double batch
            x_i = x_i.repeat(2, 1, 1, 1)
            t_is = t_is.repeat(2, 1, 1, 1)

            z = torch.randn(n_sample, *size).to(device) if i > 1 else 0

            eps = self.net(x_i, c_i, t_is, context_mask)
            eps1 = eps[:n_sample]
            eps2 = eps[n_sample:]
            eps = (1 + guide_w) * eps1 - guide_w * eps2
            x_i = x_i[:n_sample]
            x_i = (
                self.oneover_sqrta[i] * (x_i - eps * self.mab_over_sqrtmab[i]) +
                self.sqrt_beta_t[i] * z
            )
            if i % 20 == 0 or i == self.n_T or i < 8:
                x_i_store.append(x_i.detach().cpu().numpy())

        x_i_store = np.array(x_i_store)
        return x_i, x_i_store

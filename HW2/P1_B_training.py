import random
from pathlib import Path

import torch
import torch.nn as nn
from pytorch_fid import fid_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from P1_B_model import Discriminator, Generator
from P1_dataloader import p1_dataset


# https://github.com/LynnHo/DCGAN-LSGAN-WGAN-GP-DRAGAN-Pytorch
def get_FID(device, generator, out_dir, eval_noise):
    batch_size = 100
    idx = 0
    with torch.no_grad():
        gen_imgs = generator(eval_noise).cpu()
        gen_imgs = UnNormalize(gen_imgs)
        for img in gen_imgs:
            save_image(img, out_dir / f'{idx}.png')
            idx += 1
    writer.add_images('GAN results', gen_imgs, epoch)
    if epoch <= 60:
        return 10e10
    FID = fid_score.calculate_fid_given_paths(
        [str(out_dir), 'hw2_data/face/val'],
        batch_size=eval_noise.size(0),
        device=device,
        dims=2048,
        num_workers=8,
    )
    return FID


def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


mean = [0.5, 0.5, 0.5]  # [0.5696, 0.4315, 0.3593]
std = [0.5, 0.5, 0.5]  # [0.2513, 0.2157, 0.1997]
UnNormalize = transforms.Normalize(
    mean=[-u / s for u, s in zip(mean, std)],
    std=[1 / s for s in std],
)

train_set = p1_dataset(
    root='./hw2_data/face/train',
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)

batch_size = 128
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)

num_epochs = 9999
num_critic = 5
lr = 1e-5
z_dim = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if device == torch.device('cuda'):
    torch.backends.cudnn.benchmark = True

G = Generator(latent_size=z_dim, n_featuremap=128).to(device)
D = Discriminator(n_featuremap=128).to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

print(
    f"G #param: {sum(p.numel() for p in G.parameters() if p.requires_grad) / 10e6}M")
print(
    f"D #param: {sum(p.numel() for p in D.parameters() if p.requires_grad) / 10e6}M")
eval_noise = torch.randn(64, z_dim, 1, 1, device=device)


ckpt_path = Path('./P1_B_ckpt')
tb_path = Path('./P1_B_tb')
out_path = Path('./P1_B_out')
rm_tree(ckpt_path)
rm_tree(tb_path)
rm_tree(out_path)
writer = SummaryWriter(tb_path)
ckpt_path.mkdir(exist_ok=True)
out_path.mkdir(exist_ok=True)
best_epoch = -1
best_FID = 10e10
iters = 0
for epoch in range(num_epochs):
    for x_real in tqdm(train_loader):
        x_real = x_real.to(device, non_blocking=True)
        iters += 1

        # Train D
        D_optimizer.zero_grad()
        z = torch.rand(x_real.shape[0], z_dim, 1, 1, device=device)
        x_fake = G(z)
        real_D_logits = D(x_real)
        real_D_logits = D(x_real)
        # Detach since only updating D
        fake_D_logits = D(x_fake.detach())

        real_D_loss = -real_D_logits.mean()  # maximize real
        fake_D_loss = fake_D_logits.mean()  # minimize fake
        gp = D.calc_gp(x_real, x_fake)
        D_loss = (real_D_loss + fake_D_loss) + 10 * gp

        D_loss.backward()
        D_optimizer.step()
        writer.add_scalar('Train/W_dis', -real_D_loss -
                          fake_D_loss, global_step=iters)  # W_dis = real_center - fake_center
        writer.add_scalar('Train/GP', 10 * gp, global_step=iters)
        writer.add_scalars('Train', {
            'Real_center': -real_D_loss,
            'Fake_center': fake_D_loss,
        }, global_step=iters)

        if iters % num_critic == 0:
            # Train G
            G_optimizer.zero_grad()
            z = torch.rand(x_real.shape[0], z_dim, 1, 1, device=device)
            x_fake = G(z)
            fake_D_logits = D(x_fake)
            G_loss = -fake_D_logits.mean()  # maximize fake

            G_loss.backward()
            G_optimizer.step()

    FID = get_FID(device=device, generator=G,
                  out_dir=out_path, eval_noise=eval_noise)
    if FID < best_FID:
        best_FID = FID
        best_epoch = epoch
        torch.save(G.state_dict(), ckpt_path / f"best_G.pth")
        torch.save(G.state_dict(), ckpt_path / f"best_D.pth")
        print(f"[NEW] EPOCH {epoch} BEST FID: {FID}")
    else:
        print(f"[BAD] EPOCH {epoch} FID: {FID}, BEST FID: {best_FID}")

print(f"[RST] EPOCH {best_epoch}, best FID: {best_FID}")

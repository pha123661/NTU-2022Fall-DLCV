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
    generator.eval()
    idx = 0
    with torch.no_grad():
        gen_imgs = generator(eval_noise).cpu()
        gen_imgs = invTrans(gen_imgs)
        for img in gen_imgs:
            save_image(img, out_dir / f'{idx}.png')
            idx += 1
    generator.train()
    FID = fid_score.calculate_fid_given_paths(
        [str(out_dir), 'hw2_data/face/val'],
        batch_size=batch_size,
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


mean = [0.5696, 0.4315, 0.3593]  # calculated on training set
std = [0.2513, 0.2157, 0.1997]
invTrans = transforms.Normalize(
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

batch_size = 2048
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)

num_epochs = 1500
num_TrainD = 1
lr = 2e-3
z_dim = 128

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

G = Generator(latent_size=z_dim).to(device)
D = Discriminator().to(device)
G_optimizer = torch.optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.999))
D_optimizer = torch.optim.Adam(D.parameters(), lr=lr, betas=(0.5, 0.999))

# plot every epoch
plot_noise = torch.randn(64, z_dim, 1, 1, device=device)
eval_noise = torch.randn(batch_size, z_dim, 1, 1, device=device)


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
count_D = 0

for epoch in range(num_epochs):
    for x_real in tqdm(train_loader):
        x_real = x_real.to(device, non_blocking=True)
        z = torch.rand(x_real.shape[0], z_dim, 1, 1, device=device)
        x_fake = G(z)

        # Train D
        real_D_logits = D(x_real)
        fake_D_logits = D(x_fake.detach())  # Detach since only updating D

        real_D_loss = -real_D_logits.mean()
        fake_D_loss = fake_D_logits.mean()
        gp = D.calc_gp(x_real, x_fake)
        D_loss = (real_D_loss + fake_D_loss) + 10 * gp

        D_loss.backward()
        D_optimizer.step()
        D_optimizer.zero_grad()

        count_D += 1
        writer.add_scalar('Train/W_dis', fake_D_loss +
                          real_D_loss, global_step=epoch)
        writer.add_scalars('Train', {
            'Real center': real_D_loss,
            'Fake center': fake_D_loss,
        }, global_step=epoch)

        # Train G
        if count_D % num_TrainD != 0:
            continue
        fake_D_logits = D(x_fake)
        G_loss = -fake_D_logits.mean()

        G_loss.backward()
        G_optimizer.step()
        G_optimizer.zero_grad()

    with torch.no_grad():
        plot_img = G(plot_noise).detach().cpu()
        plot_img = invTrans(plot_img)
        grid = make_grid(plot_img, padding=2)
        writer.add_image('GAN results', grid, epoch)
    FID = get_FID(device=device, generator=G,
                  out_dir=out_path, eval_noise=eval_noise)
    if FID <= best_FID:
        best_FID = FID
        best_epoch = epoch
        torch.save(G.state_dict(), ckpt_path / f"best_G.pth")
        torch.save(G.state_dict(), ckpt_path / f"best_D.pth")
        print(f"[NEW] EPOCH {epoch} BEST FID: {FID}")
    else:
        print(f"[BAD] EPOCH {epoch} FID: {FID}, BEST FID: {best_FID}")

print(f"[RST] EPOCH {best_epoch}, best FID: {best_FID}")

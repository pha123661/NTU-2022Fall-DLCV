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

from P1_A_model import DCGAN_D, DCGAN_G
from P1_dataloader import p1_dataset


def get_FID(device, generator, out_dir, eval_noise):
    batch_size = 100
    generator.eval()
    idx = 0
    with torch.no_grad():
        gen_imgs = generator(eval_noise)
        gen_imgs = UnNormalize(gen_imgs)
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


# [0.5696, 0.4315, 0.3593]  # calculated on training set
mean = [0.5, 0.5, 0.5]
std = [0.5, 0.5, 0.5]

train_set = p1_dataset(
    root='./hw2_data/face/train',
    transform=transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)
UnNormalize = transforms.Normalize(
    mean=[-u / s for u, s in zip(mean, std)],
    std=[1 / s for s in std],
)

batch_size = 128
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)


num_epochs = 150
lr = 2e-4
ckpt_path = Path('./P1_A_ckpt')
tb_path = Path('./P1_A_tb')
out_path = Path('./P1_A_out')

rm_tree(ckpt_path)
rm_tree(tb_path)
rm_tree(out_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCELoss()  # take y as 1 or 0 <-> choosing logx or log(1-x) in BCE

model_G = DCGAN_G().to(device)  # z.shape = (b, 100, 1, 1)
model_D = DCGAN_D().to(device)

optim_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))

writer = SummaryWriter(tb_path)

# plot every K iterations
plot_noise = torch.randn(64, 100, 1, 1, device=device)
eval_noise = torch.randn(batch_size, 100, 1, 1, device=device)
iters = 0
ckpt_path.mkdir(exist_ok=True)
out_path.mkdir(exist_ok=True)
best_epoch = -1
best_FID = 10e10
for epoch in range(1, num_epochs + 1):
    for real_imgs in tqdm(train_loader):
        '''
        Update Discriminator:
        maximizes log(D(x)) + log(1 - D(G(z)))
        '''
        optim_D.zero_grad()
        # train with all real batch (GAN hack)
        real_imgs = real_imgs.to(device)
        batch_size = real_imgs.shape[0]

        out_D_real = model_D(real_imgs).flatten()
        loss_D_real = loss_fn(
            out_D_real,
            torch.ones(batch_size, dtype=torch.float32, device=device)
        )
        L_D = out_D_real.mean().item()
        loss_D_real.backward()

        # train with all fake batch
        z = torch.randn(batch_size, 100, 1, 1, device=device)  # normal dist.
        syn_imgs = model_G(z)

        # detach since we are updating D
        out_D_syn = model_D(syn_imgs.detach()).flatten()
        loss_D_syn = loss_fn(
            out_D_syn,
            torch.zeros(batch_size, dtype=torch.float32, device=device)
        )
        L_G_1 = out_D_syn.mean().item()
        loss_D_syn.backward()
        optim_D.step()
        loss_D = (loss_D_real + loss_D_syn).item()

        '''
        Update Generator:
        ~minimizes log(1 - D(G(z)))~
        -> maximizes log(D(G(z)))
        '''
        optim_G.zero_grad()
        out_G = model_D(syn_imgs).flatten()
        loss_G = loss_fn(
            out_G,
            torch.ones(batch_size, dtype=torch.float32, device=device),
        )
        L_G_2 = out_G.mean().item()
        loss_G.backward()
        optim_G.step()
        loss_G = loss_G.item()

        writer.add_scalar('loss/D', loss_D, global_step=iters)
        writer.add_scalar('loss/G', loss_G, global_step=iters)

        writer.add_scalar('D/D(x)', L_D, global_step=iters)
        writer.add_scalar('D/D(G(z)) 1', L_G_1, global_step=iters)
        writer.add_scalar('D/D(G(z)) 2', L_G_2, global_step=iters)
        iters += 1

    with torch.no_grad():
        plot_img = model_G(plot_noise).detach().cpu()
        plot_img = UnNormalize(plot_img)
        grid = make_grid(plot_img, padding=2)
        writer.add_image('GAN results', grid, epoch)

    FID = get_FID(device=device, generator=model_G,
                  out_dir=out_path, eval_noise=eval_noise)
    if FID <= best_FID:
        best_FID = FID
        best_epoch = epoch
        print(f"[NEW] EPOCH {epoch} BEST FID: {FID}")
        torch.save(model_G.state_dict(), ckpt_path / "best_G.pth")
        torch.save(model_D.state_dict(), ckpt_path / "best_D.pth")
    else:
        print(f"[BAD] EPOCH {epoch} FID: {FID}, BEST FID: {best_FID}")

print(f"[RST] EPOCH {best_epoch}, best FID: {best_FID}")

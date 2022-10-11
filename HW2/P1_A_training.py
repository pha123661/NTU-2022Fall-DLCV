from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from P1_dataloader import p1_dataset
from P1_model import DCGAN_D, DCGAN_G


def rm_tree(pth: Path):
    for child in pth.iterdir():
        if child.is_file():
            child.unlink()
        else:
            rm_tree(child)
    pth.rmdir()


mean = [0.5696, 0.4315, 0.3593]  # calculated on training set
std = [0.2513, 0.2157, 0.1997]

train_set = p1_dataset(
    root='./hw2_data/face/train',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)
val_set = p1_dataset(
    root='./hw2_data/face/val',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
)

batch_size = 128
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)
valid_loader = DataLoader(
    val_set, batch_size=2 * batch_size, shuffle=False, num_workers=6)


num_epochs = 10
lr = 2e-4
ckpt_path = Path('./P1_A_ckpt')
tb_path = Path('./P1_A_tb')

rm_tree(ckpt_path)
rm_tree((tb_path))

plot_K_iters = 100
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCELoss()  # take y as 1 or 0 <-> choosing logx or log(1-x) in BCE

model_G = DCGAN_G().to(device)  # z.shape = (b, 100, 1, 1)
model_D = DCGAN_D().to(device)

optim_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))

writer = SummaryWriter(tb_path)

# plot every K iterations
plot_noise = torch.randn(64, 100, 1, 1, device=device)
iters = 0
ckpt_path.mkdir(exist_ok=True)
for epoch in range(1, num_epochs + 1):
    pbar = tqdm(train_loader)
    for real_imgs in pbar:
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

        pbar.set_description(
            f"D(x): {L_D:.4f}, D(G(z)): {L_G_1:.4f}, {L_G_2:.4f}")

        if (iters % plot_K_iters) == 0:
            with torch.no_grad():
                plot_img = model_G(plot_noise).detach().cpu()
                grid = make_grid(plot_img, padding=2, normalize=True)
                writer.add_image('Generated results', grid, iters)
            torch.save(model_G.state_dict(), ckpt_path / f"{iters}_G.pth")
            torch.save(model_D.state_dict(), ckpt_path / f"{iters}_D.pth")
            print('new result saved!')
        iters += 1

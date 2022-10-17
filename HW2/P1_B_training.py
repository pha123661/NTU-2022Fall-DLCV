from pathlib import Path

import torch
import torch.nn as nn
from pytorch_fid import fid_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from P1_B_model import DCGAN_G, SNGAN_D
from P1_dataloader import p1_dataset

FID_cool_down = 10


def CallBack(device, generator, out_dir, eval_noise):
    generator.eval()
    idx = 0
    with torch.no_grad():
        gen_imgs = generator(eval_noise)
        gen_imgs = UnNormalize(gen_imgs)
        for img in gen_imgs:
            save_image(img, out_dir / f'{idx}.png')
            idx += 1
        writer.add_images('GAN results', gen_imgs, epoch)
    generator.train()

    global FID_cool_down
    if FID_cool_down > 0:
        FID_cool_down -= 1
        return -1

    FID = fid_score.calculate_fid_given_paths(
        [str(out_dir), 'hw2_data/face/val'],
        batch_size=gen_imgs.size(0),
        device=device,
        dims=2048,
        num_workers=8,
    )
    if FID > 50:
        FID_cool_down += 10
    return FID


def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


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


num_epochs = 99999
lr = 2e-4
ckpt_path = Path('./P1_B_ckpt')
tb_path = Path('./P1_B_tb')
out_path = Path('./P1_B_out')

rm_tree(ckpt_path)
rm_tree(tb_path)
rm_tree(out_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loss_fn = nn.BCELoss()  # take y as 1 or 0 <-> choosing logx or log(1-x) in BCE

model_G = DCGAN_G().to(device)  # z.shape = (b, 100, 1, 1)
model_D = SNGAN_D().to(device)

optim_G = torch.optim.Adam(model_G.parameters(), lr=lr, betas=(0.5, 0.999))
optim_D = torch.optim.Adam(model_D.parameters(), lr=lr, betas=(0.5, 0.999))

writer = SummaryWriter(tb_path)

eval_noise = torch.randn(64, 100, 1, 1, device=device)
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
        loss_G.backward()
        optim_G.step()
        loss_G = loss_G.item()
        writer.add_scalars(
            'Loss', {'D': loss_D, 'G': loss_G}, global_step=iters)

        iters += 1

    FID = CallBack(device=device, generator=model_G,
                   out_dir=out_path, eval_noise=eval_noise)
    if FID == -1:
        continue
    writer.add_scalar('FID', FID, global_step=epoch)
    if FID <= best_FID:
        best_FID = FID
        best_epoch = epoch
        print(f"[NEW] EPOCH {epoch} BEST FID: {FID}")
        torch.save(model_G.state_dict(), ckpt_path / "best_G.pth")
        torch.save(model_D.state_dict(), ckpt_path / "best_D.pth")
    else:
        print(
            f"EPOCH {epoch} FID: {FID}, BEST FID: {best_FID} @ EPOCH {best_epoch}")

print(f"[RST] best FID: {best_FID} @ EPOCH {best_epoch}")

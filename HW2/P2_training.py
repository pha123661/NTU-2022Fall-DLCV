from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid
from tqdm import tqdm

from digit_dataloader import digit_dataset
from P2_model import DDPM_framework, Unet


# https://github.com/TeaPearce/Conditional_Diffusion_MNIST
def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


# mean, std = [0.4632, 0.4669, 0.4195], [0.1979, 0.1845, 0.2082]
train_set = digit_dataset(
    root='hw2_data/digits/mnistm/data/',
    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ]),
    label_csv=['hw2_data/digits/mnistm/train.csv',
               'hw2_data/digits/mnistm/val.csv']
)

batch_size = 256
train_loader = DataLoader(
    train_set, batch_size=batch_size, shuffle=True, num_workers=6)

num_epochs = 100
n_T = 500
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 1e-4
n_features = 128
ckpt_path = Path('./P2_ckpt')
tb_path = Path('./P2_tb')

rm_tree(ckpt_path)
rm_tree(tb_path)

ckpt_path.mkdir(exist_ok=True)
tb_path.mkdir(exist_ok=True)

ddpm = DDPM_framework(
    network=Unet(
        in_chans=3,
        n_features=n_features,
        n_classes=10
    ),
    betas=(1e-4, 0.02),
    n_T=n_T,
    device=device,
    drop_prob=0.1
).to(device)
optim = torch.optim.Adam(ddpm.parameters(), lr=lr)
scaler = torch.cuda.amp.GradScaler()
writer = SummaryWriter(tb_path)

for epoch in range(num_epochs):
    ddpm.train()

    # linear lr decay
    optim.param_groups[0]['lr'] = lr * (1 - epoch / num_epochs)

    for x, c in tqdm(train_loader):
        with torch.autocast(device_type='cuda' if device != 'cpu' else 'cpu', dtype=torch.float16):
            x = x.to(device, non_blocking=True)
            c = c.to(device, non_blocking=True)
            loss = ddpm(x, c)
        scaler.scale(loss).backward()
        scaler.step(optim)  # replaces optim.step()
        scaler.update()
        optim.zero_grad()

    ddpm.eval()
    with torch.no_grad():
        n_samples = 30
        for gw in [0, 0.5, 2]:
            x_gen, x_gen_store = ddpm.sample(
                n_samples, (3, 28, 28), device, guide_w=gw)
            grid = make_grid(x_gen * -1 + 1, nrow=3)
            writer.add_image(f'DDPM results/w={gw:.1f}', grid, epoch)
            grid = make_grid(x_gen, nrow=3)
            writer.add_image(f'DDPM results wo inv/w={gw:.1f}', grid, epoch)

    torch.save(ddpm.state_dict(), ckpt_path / f"{epoch}_ddpm.pth")

import random
import sys
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from P2_model import DDPM_framework, Unet

random.seed(0)
torch.manual_seed(0)

ckpt_dir = Path('./P2_ckpt/98_ddpm.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = DDPM_framework(
    network=Unet(
        in_chans=3,
        n_features=128,
        n_classes=10
    ),
    betas=(1e-4, 0.02),
    n_T=500,
    device=device,
    drop_prob=0.1
).to(device)
model.load_state_dict(torch.load(ckpt_dir, map_location=device))
model.eval()

with torch.no_grad():
    x_i, x_i_store = model.sample(100, size=(
        3, 28, 28), device=device, guide_w=2)
    x_i = x_i.reshape(10, 10, 3, 28, 28)
    x_i = torch.transpose(x_i, 0, 1)
    x_i = x_i.reshape(-1, 3, 28, 28)
    save_image(x_i, './P2_plot/100_samples.png', nrow=10)
    save_image(torch.tensor(x_i_store[:, 0, ...].reshape(
        32, 3, 28, 28)), './P2_plot/first_sample_progress.png')

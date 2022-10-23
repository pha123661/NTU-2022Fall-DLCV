import random
import sys
from pathlib import Path

import torch
from torchvision.utils import save_image

from P2_model import DDPM_framework, Unet

random.seed(0)
torch.manual_seed(0)

out_dir = Path(sys.argv[1])
try:
    out_dir.mkdir(exist_ok=True, parents=True)
except:
    pass
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

for class_idx in range(10):
    gened_count = 0
    with torch.no_grad():
        x_i, _ = model.class_gen(100, size=(3, 28, 28),
                                 device=device, class_idx=class_idx, guide_w=2)
    for image in x_i:
        gened_count += 1
        save_image(image, out_dir / f'{class_idx}_{gened_count}.png')

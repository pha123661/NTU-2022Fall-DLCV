import argparse
import pathlib

import torch
from torchvision.utils import save_image

from P1_model import DCGAN_G

parser = argparse.ArgumentParser()
parser.add_argument(
    "-w", "--weight",
    help="path to generator weight",
    type=pathlib.Path
)
parser.add_argument(
    "-o", "--out_dir",
    help="output directory",
    type=pathlib.Path,
    default='./P1_A_out',
)
parser.add_argument(
    "-d", "--device",
    help="device",
    type=torch.device,
    default='cuda' if torch.cuda.is_available() else 'cpu',
)
args = parser.parse_args()

args.out_dir.mkdir(exist_ok=True)

batch_size = 100
model = DCGAN_G().to(args.device)
model.load_state_dict(torch.load(args.weight, map_location=args.device))

idx = 0
for i in range(1000 // batch_size):
    z = torch.randn(batch_size, 100, 1, 1, device=args.device)
    gen_imgs = model(z)
    for img in gen_imgs:
        save_image(img, args.out_dir / f'{idx}.png', normalize=True)
        idx += 1

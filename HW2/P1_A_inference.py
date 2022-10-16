import argparse
import pathlib
import random

import torch
from torchvision.utils import save_image

from P1_A_model import DCGAN_G

# python3 P1_A_inference.py  -w ./P1_A_ckpt/best_G.pth -o ./P1_A_out
# pytorch-fid ./P1_A_out/ ./hw2_data/face/val/
# python3 face_recog.py --image_dir ./P1_A_out


def main(device, weight, out_dir):
    # 7: 28.793507172455804, 90.90
    seed = 7
    random.seed(seed)
    torch.manual_seed(seed)

    batch_size = 100
    model = DCGAN_G().to(device)
    model.load_state_dict(torch.load(weight, map_location=device))

    idx = 0
    for _ in range(1000 // batch_size):
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        gen_imgs = model(z)
        for img in gen_imgs:
            save_image(img, out_dir / f'{idx}.png', normalize=True)
            idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weight",
        help="path to generator weight",
        type=pathlib.Path,
        default='./P1_A_ckpt/best_G.pth'
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
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    args.out_dir.mkdir(exist_ok=True)
    main(args.device, args.weight, args.out_dir)

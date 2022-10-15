import argparse
import pathlib
import random

import torch
from torchvision.utils import save_image

from P1_A_model import DCGAN_G


def main(device, weight, out_dir):
    random.seed(0)
    torch.manual_seed(0)

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

    # FID = fid_score.calculate_fid_given_paths(
    #     [str(out_dir), 'hw2_data/face/val'],
    #     batch_size=batch_size,
    #     device=device,
    #     dims=2048,
    #     num_workers=8,
    # )
    # print(f"FID: {FID}")
    # return FID


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weight",
        help="path to generator weight",
        type=pathlib.Path,
        required=True,
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

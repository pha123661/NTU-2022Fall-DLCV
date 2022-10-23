import argparse
import pathlib
import random

import torch
from torchvision import transforms
from torchvision.utils import save_image

from P1_B_model import DCGAN_G

# python3 P1_B_inference.py  -w ./P1_B_ckpt/best_G.pth
# pytorch-fid ./P1_B_out/ ./hw2_data/face/val/
# python3 face_recog.py --image_dir ./P1_B_out


def main(device, weight, out_dir):
    # FID = 25.9896166834331, ACC = 91.100%
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]
    UnNormalize = transforms.Normalize(
        mean=[-u / s for u, s in zip(mean, std)],
        std=[1 / s for s in std],
    )

    batch_size = 100
    model = DCGAN_G().to(device)
    model.load_state_dict(torch.load(weight, map_location=device))

    idx = 0
    for _ in range(1000 // batch_size):
        z = torch.randn(batch_size, 100, 1, 1, device=device)
        with torch.no_grad():
            gen_imgs = model(z)
        gen_imgs = UnNormalize(gen_imgs)
        for img in gen_imgs:
            save_image(img, out_dir / f'{idx}.png')
            idx += 1


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-w", "--weight",
        help="path to generator weight",
        type=pathlib.Path,
        default='./P1_B_ckpt/best_G.pth'
    )
    parser.add_argument(
        "-o", "--out_dir",
        help="output directory",
        type=pathlib.Path,
        default='./P1_B_out',
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
    try:
        args.out_dir.mkdir(exist_ok=True, parents=True)
    except:
        pass
    main(args.device, args.weight, args.out_dir)

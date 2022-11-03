import argparse
import json
import pathlib
import shutil

import torch
from language_evaluation import CocoEvaluator
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ICDataset import ICDataset
from P2_model import ImageCaptioningTransformer


def main(args):
    tokenizer = Tokenizer.from_file(args.tokenizer)
    transform = create_transform(**resolve_data_config({}, model=args.model))
    valid_set = ICDataset(
        image_dir=args.valid_image_dir,
        json_file=args.valid_info,
        transform=transform,
        tokenizer=tokenizer
    )
    valid_loader = DataLoader(valid_set, 2 * args.batch_size,
                              collate_fn=valid_set.collate_fn,
                              shuffle=False,
                              num_workers=4,
                              pin_memory=True)
    Transformer = ImageCaptioningTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        encoder=args.model,
        num_layers=4,
        nhead=12,
        d_model=768,
    )


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--valid_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--valid_info', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output Path
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Training args
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args)

import argparse
import glob
import json
import os
import pathlib

import torch
from language_evaluation import CocoEvaluator
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from P2_model import ImageCaptioningTransformer


class Image_dataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        self.Transform = transform
        self.Image_names = [p for p in root.glob("*")]

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx]).convert('RGB')
        img = self.Transform(img)

        return img, os.path.splitext(os.path.basename(self.Image_names[idx]))[0]

    def __len__(self):
        return len(self.Image_names)


def main(args):
    tokenizer = Tokenizer.from_file(args.tokenizer)
    transform = create_transform(**resolve_data_config({}, model=args.model))
    valid_set = Image_dataset(
        root=args.valid_image_dir,
        transform=transform,
    )
    # valid_loader = DataLoader(valid_set, 2 * args.batch_size,
    #                           collate_fn=valid_set.collate_fn,
    #                           shuffle=False,
    #                           num_workers=4,
    #                           pin_memory=True)
    Transformer = ImageCaptioningTransformer(
        **json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
    ).to(args.device)
    Transformer.load_state_dict(torch.load(
        args.ckpt_dir / "Best_model.pth", map_location=args.device))
    Transformer.eval()

    pred = dict()
    for data, name in tqdm(valid_set):
        output_ids = Transformer.greedy_search(data.to(args.device))
        sentence = tokenizer.decode(output_ids)
        pred[name] = sentence

    json.dump(pred, args.output_json.open(mode='w'), indent=4)


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--valid_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    # parser.add_argument('--valid_info', type=pathlib.Path,
    #                     default='hw3_data/p2_data/val.json')
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Output Path
    parser.add_argument('--output_json', type=pathlib.Path)

    # Training args
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.output_json.parent.mkdir(exist_ok=True, parents=True)
    main(args)

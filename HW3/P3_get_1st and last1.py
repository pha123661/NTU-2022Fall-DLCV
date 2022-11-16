import argparse
import json
import os
import pathlib
from collections import defaultdict

import clip
import language_evaluation
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import Dataset
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
    # config = json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
    # tokenizer = Tokenizer.from_file(args.tokenizer)
    # transform = create_transform(
    #     **resolve_data_config({}, model=timm.create_model(config['encoder'], pretrained=True, num_classes=0)))
    # valid_set = Image_dataset(
    #     root=args.image_dir,
    #     transform=transform,
    # )
    # Model = ImageCaptioningTransformer(**config)
    # Model.load_state_dict(torch.load(
    #     args.ckpt_dir / "Best_model.pth", map_location=args.device))
    # Model.to(args.device)
    # Model.eval()

    # preds = dict()
    # for data, name in tqdm(valid_set):
    #     output_ids = Model.greedy_search(data.to(args.device))
    #     sentence = tokenizer.decode(output_ids)
    #     preds[name] = sentence
    preds = json.load(open("pred.json", mode='r'))

    # CIDEr score
    evaluator = language_evaluation.CocoEvaluator(coco_types=['CIDEr'])
    info = json.load(args.info_json.open(mode='r'))
    annotations = defaultdict(list)
    for data in info['annotations']:
        annotations[data['image_id']].append(data['caption'])
    img2id = {os.path.splitext(data['file_name'])[0]: data['id']
              for data in info['images']}

    min_value = 10e10
    min_name = None
    max_value = 0
    max_name = None
    # CLIP score
    model, image_process = clip.load('ViT-B/32', device=args.device)

    for image_name, text in tqdm(preds.items()):
        image = Image.open(
            args.image_dir / f"{image_name}.jpg").convert('RGB')
        image = image_process(image).unsqueeze(0).to(args.device)
        text = clip.tokenize(text).to(args.device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        sim = image_features @ text_features.T
        score = 2.5 * max(sim.item(), 0)
        if score < min_value:
            min_value = score
            min_name = image_name
        if score > max_value:
            max_value = score
            max_name = image_name

    print(f"Min {min_name}, {min_value}")
    print(f"Max {max_name}, {max_value}")


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Validation args
    parser.add_argument('--info_json', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args)

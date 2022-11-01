import argparse
import json
import pathlib

import clip
import torch
from PIL import Image


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p1_data/val')
    parser.add_argument('--id2label', type=pathlib.Path,
                        default='hw3_data/p1_data/id2label.json')
    parser.add_argument('--outcsv', type=pathlib.Path,
                        default='./output_p1/pred.csv')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='RN50x4')
    args = parser.parse_args()
    return args


def main(args):
    image_names = args.image_dir.glob('*.png')
    id2label = json.load(args.id2label.open(mode='r'))
    labels = [v for k, v in id2label.items()]
    model, image_process = clip.load(args.model, device=args.device)

    for image_name in image_names:
        image = Image.open(image_name).convert('RGB')
        image = image_process(image).unsqueeze(0).to(args.device)
        text = torch.cat(
            [clip.tokenize(f'a photo of a {c}') for c in labels]).to(args.device)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        print(image_features.shape, text_features.shape)


if __name__ == '__main__':
    args = parse()
    try:
        args.outcsv.parent.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print(e)
    main(args)

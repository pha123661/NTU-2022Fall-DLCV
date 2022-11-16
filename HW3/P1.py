import argparse
import csv
import json
import os
import pathlib

import clip
import torch
from PIL import Image
from tqdm.auto import tqdm


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p1_data/val')
    parser.add_argument('--id2label', type=pathlib.Path,
                        default='hw3_data/p1_data/id2label.json')
    parser.add_argument('--out_csv', type=pathlib.Path,
                        default='./output_p1/pred.csv')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='ViT-L-14')
    parser.add_argument("--do_eval", action='store_true')
    args = parser.parse_args()
    return args


def main(args):
    image_names = [i for i in args.image_dir.glob('*')]
    id2label = json.load(args.id2label.open(mode='r'))
    labels = [v for k, v in id2label.items()]
    model, image_process = clip.load(args.model, device=args.device)

    ALL_NAMES = []
    ALL_LABELS = []
    for image_name in tqdm(image_names):
        image = Image.open(image_name).convert('RGB')
        image = image_process(image).unsqueeze(0).to(args.device)
        text = torch.cat(
            [clip.tokenize(f'a photo of a {c}') for c in labels]).to(args.device)

        with torch.no_grad():
            image_features = model.encode_image(image)  # (1, 512)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features = model.encode_text(text)  # (50, 512)
            text_features /= text_features.norm(dim=-1, keepdim=True)

        # sim[i, j] = i-th image to j-th text
        sim = (image_features @ text_features.T).softmax(dim=-1)
        label = sim[0].argmax()
        ALL_NAMES.append(image_name.name)
        ALL_LABELS.append(label.item())

    with args.out_csv.open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('filename', 'label'))
        for data in zip(ALL_NAMES, ALL_LABELS):
            writer.writerow(data)

    if args.do_eval:
        corr = 0
        for n, l in zip(ALL_NAMES, ALL_LABELS):
            if int(n.split('_')[0]) == l:
                corr += 1
        print(f'accuracy: {corr / len(ALL_NAMES)}')


if __name__ == '__main__':
    args = parse()
    try:
        args.out_csv.parent.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print(e)
    main(args)

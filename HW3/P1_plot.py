import argparse
import json
import pathlib
import random

import clip
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from PIL import Image


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p1_data/val')
    parser.add_argument('--id2label', type=pathlib.Path,
                        default='hw3_data/p1_data/id2label.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--model', default='P1_ckpt/ViT-L-14.pt')
    args = parser.parse_args()
    return args


def main(args):
    image_names = [i for i in args.image_dir.glob('*')]
    id2label = json.load(args.id2label.open(mode='r'))
    labels = np.array([v for k, v in id2label.items()])

    model, image_process = clip.load(args.model, device=args.device)
    for idx in range(1, 4):
        image_name = random.choice(image_names)
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
        corr_prob = sim[0][int(image_name.name.split('_')[0])]
        sim, indices = sim[0].topk(5)
        sim = sim.cpu().numpy() * 100
        indices = indices.cpu().numpy()

        image = Image.open(image_name).convert('RGB')
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title(
            f"Correct label: {id2label[image_name.name.split('_')[0]]}")

        print(sim)
        sns.barplot(
            x=sim,
            y=[f'a photo of a {c}' for c in labels[indices]],
            ax=axes[1],
        )
        axes[1].tick_params(axis="y", direction="in", pad=-150)
        axes[1].set(xlim=[0, 3],
                    title=f"{id2label[image_name.name.split('_')[0]]} probability: {corr_prob*100:.2f}%")

        plt.savefig(f"P1_plot/{idx}")


if __name__ == '__main__':
    args = parse()
    main(args)

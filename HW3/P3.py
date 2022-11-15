import argparse
import json
import os
import pathlib

import torch
from PIL import Image
from tokenizers import Tokenizer
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
from torchvision.transforms.functional import resize
import matplotlib.pyplot as plt

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


def register_attention_hook(model, features, feature_sizes):
    def hook_decoder(module, ins, outs):
        features.append(outs[1].detach().cpu())
    handle_decoder = model.decoder.layers[-1].multihead_attn.register_forward_hook(
        hook_decoder)
    return [handle_decoder]


def vis_atten_map(atten_mat, ids, feature_size, image_fn, image_path):
    print(atten_mat.shape)
    nrows = len(ids) // 5 if len(ids) % 5 == 0 else len(ids) // 5 + 1
    ncols = 5
    fig, ax = plt.subplots(ncols=ncols, nrows=nrows, figsize=(16, 8))
    feature_size = (14, 14)  # H/patch size, W/patch size = feature size
    for i, id in enumerate(ids):
        attn_vector = atten_mat[:, i - 1, 1:]
        attn_map = torch.reshape(attn_vector, feature_size)
        attn_map -= torch.min(attn_map)
        attn_map /= torch.max(attn_map)
        # print(torch.min(attn_map), torch.max(attn_map))
        # print(attn_map.size())
        im = Image.open(image_path)
        size = im.size
        mask = resize(attn_map.unsqueeze(0), [size[1], size[0]]).squeeze(0)
        mask = np.uint8(mask * 255)
        # print(mask.shape)
        ax[i // 5][i % 5].imshow(im)
        if i == 0:
            ax[i // 5][i % 5].set_title('<SOS>')
        elif i == len(ids) - 1:
            ax[i // 5][i % 5].set_title('<EOS>')
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        else:
            ax[i // 5][i % 5].set_title(id)
            ax[i // 5][i % 5].imshow(mask, alpha=0.7, cmap='jet')
        ax[i // 5][i % 5].axis('off')
    for i in range(len(ids), nrows * ncols):
        ax[i // 5][i % 5].axis('off')
    plt.savefig(args.output_dir / image_fn)


def main(args):
    config = json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
    tokenizer = Tokenizer.from_file(args.tokenizer)
    transform = transforms.Compose([
        transforms.Resize(
            224, interpolation=transforms.InterpolationMode.BICUBIC, max_size=None, antialias=None),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4850, 0.4560, 0.4060], std=[
                             0.2290, 0.2240, 0.2250])
    ])
    valid_set = Image_dataset(
        root=args.image_dir,
        transform=transform,
    )
    config['pretrained'] = False
    Model = ImageCaptioningTransformer(**config)
    Model.load_state_dict(torch.load(
        args.ckpt_dir / "Best_model.pth", map_location=args.device))
    Model.to(args.device)
    Model.eval()

    for data, name in valid_set:
        features, feature_sizes = [], []
        to_rm_l = register_attention_hook(Model, features, feature_sizes)
        output_ids = Model.greedy_search(data.to(args.device))
        output_tokens = ['<SOS>']
        output_tokens.extend([tokenizer.id_to_token(id) for id in output_ids])
        output_tokens.append('<EOS>')

        # visualize
        attention_matrix = features[-1]
        vis_atten_map(attention_matrix, output_tokens, feature_sizes,
                      name, (args.image_dir / name).with_suffix('.jpg'))

        for handle in to_rm_l:
            handle.remove()


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--image_dir', type=pathlib.Path,
                        default='hw3_data/p3_data/images/')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Validation args
    parser.add_argument("--output_dir", type=pathlib.Path, default="P3_plot")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.output_dir.mkdir(exist_ok=True, parents=True)
    main(args)

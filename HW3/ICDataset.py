import json
import os

import torch
from PIL import Image
from torch.utils.data import Dataset


class ICDataset(Dataset):
    def __init__(self, image_dir, json_file, transform, tokenizer, training=False) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.transform = transform
        self.tokenizer = tokenizer

        info = json.load(json_file.open(mode='r'))
        self.data = [
            {'caption': data['caption'], 'image_id': data['image_id']}
            for data in info['annotations']
        ]
        self.id2img = {data['id']: data['file_name']
                       for data in info['images']}

    def __getitem__(self, index):
        info = self.data[index]
        imgname = self.id2img[info['image_id']]
        img = Image.open(self.image_dir / imgname).convert('RGB')
        img = self.transform(img)
        return {
            'image': img,
            'caption': info['caption'],
            'filename': os.path.splitext(imgname)[0]
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):
        captions = list()
        filenames = list()
        images = list()
        for sample in samples:
            captions.append(sample['caption'])
            filenames.append(sample['filename'])
            images.append(sample['image'])

        captions = self.tokenizer.encode_batch(captions)
        input_ids = torch.tensor([c.ids for c in captions])
        images = torch.stack(images, dim=0)
        return {
            'input_ids': input_ids,
            'filenames': filenames,
            'images': images,
        }


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


if __name__ == '__main__':
    from pathlib import Path

    from torchvision import transforms
    train_set = ICDataset(
        image_dir=Path('hw3_data/p2_data/images/train'),
        json_file=Path('hw3_data/p2_data/train.json'),
        transform=transforms.ToTensor()
    )

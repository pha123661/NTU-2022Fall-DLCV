import csv
import glob
import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset


class ImageFolderDataset(Dataset):
    def __init__(self, folder_path, transform, label_csv=None):
        self.Filenames = glob.glob(os.path.join(folder_path, "*.jpg"))
        self.Transform = transform
        self.fname2label = None

        if label_csv is not None:
            self.fname2label = dict()
            with open(label_csv, 'r') as f:
                rows = csv.reader(f)
                next(rows)
                for row in rows:
                    self.fname2label[row[1]] = row[2]
            all_labels = sorted(set(self.fname2label.values()))

            self.label2idx = {v: k for k, v in enumerate(all_labels)}
            # import json
            # with open(os.path.join(os.path.dirname(label_csv), 'label2idx.json'), 'w') as f:
            #     json.dump(self.label2idx, f, indent=2)

    def __getitem__(self, idx):
        data_dict = dict()
        data_dict['img'] = self.Transform(Image.open(self.Filenames[idx]))
        data_dict['filename'] = os.path.basename(self.Filenames[idx])
        if self.fname2label:
            data_dict['label'] = self.label2idx[self.fname2label[data_dict['filename']]]

        return data_dict

    def __len__(self):
        return len(self.Filenames)


if __name__ == "__main__":
    from torchvision import transforms
    from tqdm import tqdm

    # calculates mean and std of training set
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
    ])

    dst = ImageFolderDataset(
        "hw4_data/office/train",
        train_transform,
        "hw4_data/office/train.csv",
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for data in tqdm(dst):
        x = data['img']
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))
    mean /= len(dst)
    std /= len(dst)
    print(mean, std)
    # [0.4706, 0.4495, 0.4031] [0.2176, 0.2152, 0.2150]

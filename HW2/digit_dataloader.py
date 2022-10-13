import csv
import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class digit_dataset(Dataset):
    def __init__(self, root, transform, label_csv) -> None:
        super().__init__()
        self.Transform = transform
        self.Image_names = list()
        self.Labels = list()

        with open(label_csv, mode='r') as file:
            reader = csv.reader(file)
            next(iter(reader))  # ignore first line
            for row in reader:
                self.Image_names.append(os.path.join(root, row[0]))
                self.Labels.append(int(row[1]))

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx])
        img = self.Transform(img)
        label = self.Labels[idx]
        return img, label

    def __len__(self):
        return len(self.Image_names)


if __name__ == '__main__':
    dataset = digit_dataset(
        root='hw2_data/digits/mnistm/data',
        transform=transforms.Compose([
            transforms.ToTensor(),
        ]),
        label_csv='hw2_data/digits/mnistm/train.csv'
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x, y in dataset:
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))

    mean /= len(dataset)
    std /= len(dataset)
    # tensor([0.4631, 0.4666, 0.4195]) tensor([0.1979, 0.1845, 0.2083])
    print(mean, std)

import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


class p1_dataset(Dataset):
    def __init__(self, root, transform) -> None:
        super().__init__()

        self.Transform = transform
        self.Image_names = glob.glob(os.path.join(root, "*.png"))

    def __getitem__(self, idx):
        img = Image.open(self.Image_names[idx])
        img = self.Transform(img)

        return img

    def __len__(self):
        return len(self.Image_names)


if __name__ == '__main__':
    dataset = p1_dataset(
        root='./hw2_data/face/train',
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x in dataset:
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))

    mean /= len(dataset)
    std /= len(dataset)
    # tensor([0.5696, 0.4315, 0.3593]) tensor([0.2513, 0.2157, 0.1997])
    print(mean, std)

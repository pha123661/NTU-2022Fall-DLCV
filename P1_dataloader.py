import os

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class p1_dataset(Dataset):
    def __init__(self, folder_path, transformation, train=True):
        self.Images = []
        self.Labels = []
        self.Transformation = transformation
        self.Train = train
        file_list = [file for file in os.listdir(
            folder_path) if file.endswith('.png')]
        for filename in file_list:
            self.Images.append(Image.open(os.path.join(
                folder_path, filename)).convert('RGB'))
            if train:
                label = int(filename.split("_")[0])
                self.Labels.append(label)

    def __getitem__(self, index):
        if self.Train:
            return self.Transformation(self.Images[index]), self.Labels[index]
        else:
            return self.Transformation(self.Images[index])

    def __len__(self):
        return len(self.Images)


if __name__ == "__main__":
    # calculates mean and std of training set
    train_transform = transforms.Compose([
        # transforms.Resize((32, 32)),
        # transforms.RandomCrop((28, 28)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[
        #                      0.25, 0.25, 0.25]),
    ])

    dst = p1_dataset(
        "/shared_home/r11944004/pepper_local_disk/DLCV/hw1-pha123661/hw1_data/hw1_data/p1_data/train_50", train_transform)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    for x, _ in dst:
        mean += x.mean(dim=(1, 2))
        std += x.std(dim=(1, 2))
    mean /= len(dst)
    std /= len(dst)
    print(mean, std)
    # [0.5077, 0.4813, 0.4312], [0.2000, 0.1986, 0.2034]

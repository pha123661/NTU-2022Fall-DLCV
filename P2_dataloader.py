import glob
import os

import numpy as np
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True  # some images are truncated


class p2_dataset(Dataset):
    def __init__(self, folder_path, transform=transforms.ToTensor(), train=False):

        self.Train = train
        self.Transformation = transform
        self.Images = []

        for filename in sorted(glob.glob(os.path.join(folder_path, '*.jpg'))):
            # filename is full path
            self.Images.append(Image.open(filename))

        if self.Train:
            self.Masks = np.empty(
                (len(glob.glob(os.path.join(folder_path, '*.png'))), 512, 512), dtype=np.int64)
            for i, mask_filename in enumerate(sorted(glob.glob(os.path.join(folder_path, '*.png')))):
                mask = Image.open(mask_filename)
                mask = np.array(mask)
                mask = (mask >= 128).astype(int)
                mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
                self.Masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
                self.Masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
                self.Masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
                self.Masks[i, mask == 2] = 3  # (Green: 010) Forest land
                self.Masks[i, mask == 1] = 4  # (Blue: 001) Water
                self.Masks[i, mask == 7] = 5  # (White: 111) Barren land
                self.Masks[i, mask == 0] = 6  # (Black: 000) Unknown

        else:
            self.Filenames = glob.glob(os.path.join(folder_path, '*.jpg'))
            self.Filenames.sort()
            self.Filenames = [os.path.basename(i) for i in self.Filenames]

    def __getitem__(self, idx):
        if self.Train:
            return self.Transformation(self.Images[idx]), self.Masks[idx]
        else:
            return self.Transformation(self.Images[idx]), self.Filenames[idx]

    def __len__(self):
        return len(self.Images)


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    dst = p2_dataset('./hw1_data/hw1_data/p2_data/train',
                     transform=transforms.ToTensor(), train=False)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    # for x, _ in dst:
    #     mean += x.mean(dim=(1, 2))
    #     std += x.std(dim=(1, 2))
    dst, _ = torch.utils.data.random_split(dst, [2, len(dst) - 2])
    print(type(dst), len(dst))
    for x, y in DataLoader(dst, 1):
        print(x.shape, y)

    mean /= len(dst)
    std /= len(dst)
    print(mean, std)

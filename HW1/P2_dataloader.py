import glob
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms.functional import hflip, vflip


class p2_dataset(Dataset):
    def __init__(self, path, transform, train=False, augmentation=False) -> None:
        super().__init__()

        self.Train = train
        self.Transform = transform
        self.Image_names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.Train:
            self.Mask_names = sorted(glob.glob(os.path.join(path, "*.png")))

        def identity(x): return x

        if augmentation:
            self.augmentation = [identity, hflip, identity, vflip]
        else:
            self.augmentation = [identity]

    def __getitem__(self, idx):
        if self.Train:
            img = Image.open(self.Image_names[idx])
            img = self.Transform(img)

            mask = Image.open(self.Mask_names[idx])
            mask = np.array(mask)
            mask = (mask >= 128).astype(int)
            mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]

            raw_mask = deepcopy(mask)

            mask[raw_mask == 3] = 0  # (Cyan: 011) Urban land
            mask[raw_mask == 6] = 1  # (Yellow: 110) Agriculture land
            mask[raw_mask == 5] = 2  # (Purple: 101) Rangeland
            mask[raw_mask == 2] = 3  # (Green: 010) Forest land
            mask[raw_mask == 1] = 4  # (Blue: 001) Water
            mask[raw_mask == 7] = 5  # (White: 111) Barren land
            mask[raw_mask == 0] = 6  # (Black: 000) Unknown
            mask = torch.tensor(mask)

            aug = random.choice(self.augmentation)
            img = aug(img)
            mask = aug(mask)

            return img, mask
        else:
            img = Image.open(self.Image_names[idx])
            img = self.Transform(img)

            return img, os.path.basename(self.Image_names[idx])

    def __len__(self):
        return len(self.Image_names)


if __name__ == '__main__':
    import torch
    from torchvision import transforms
    dst = p2_dataset('./hw1_data/hw1_data/p2_data/train',
                     transform=transforms.ToTensor(), train=True, augmentation=True)
    for x, y in dst:
        # print(x, y)
        pass
    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for x, _ in dst:
    #     mean += x.mean(dim=(1, 2))
    #     std += x.std(dim=(1, 2))

    # mean /= len(dst)
    # std /= len(dst)
    # print(mean, std)
    # tensor([0.4085, 0.3785, 0.2809]) tensor([0.1155, 0.0895, 0.0772])

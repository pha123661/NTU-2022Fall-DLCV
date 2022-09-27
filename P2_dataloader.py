# import glob
# import os

# import numpy as np
# from PIL import Image, ImageFile
# from torch.utils.data import DataLoader, Dataset
# from torchvision import transforms

# ImageFile.LOAD_TRUNCATED_IMAGES = True  # some images are truncated


# class p2_dataset(Dataset):
#     def __init__(self, folder_path, transform=transforms.ToTensor(), train=False):

#         self.Train = train
#         self.Transformation = transform
#         self.Images = []

#         for filename in sorted(glob.glob(os.path.join(folder_path, '*.jpg'))):
#             # filename is full path
#             self.Images.append(Image.open(filename))

#         if self.Train:
#             self.Masks = np.empty(
#                 (len(glob.glob(os.path.join(folder_path, '*.png'))), 512, 512), dtype=np.int64)
#             for i, mask_filename in enumerate(sorted(glob.glob(os.path.join(folder_path, '*.png')))):
#                 mask = Image.open(mask_filename)
#                 mask = np.array(mask)
#                 mask = (mask >= 128).astype(int)
#                 mask = 4 * mask[:, :, 0] + 2 * mask[:, :, 1] + mask[:, :, 2]
#                 self.Masks[i, mask == 3] = 0  # (Cyan: 011) Urban land
#                 self.Masks[i, mask == 6] = 1  # (Yellow: 110) Agriculture land
#                 self.Masks[i, mask == 5] = 2  # (Purple: 101) Rangeland
#                 self.Masks[i, mask == 2] = 3  # (Green: 010) Forest land
#                 self.Masks[i, mask == 1] = 4  # (Blue: 001) Water
#                 self.Masks[i, mask == 7] = 5  # (White: 111) Barren land
#                 self.Masks[i, mask == 0] = 6  # (Black: 000) Unknown

#         else:
#             self.Filenames = glob.glob(os.path.join(folder_path, '*.jpg'))
#             self.Filenames.sort()
#             self.Filenames = [os.path.basename(i) for i in self.Filenames]

#     def __getitem__(self, idx):
#         if self.Train:
#             return self.Transformation(self.Images[idx]), self.Masks[idx]
#         else:
#             return self.Transformation(self.Images[idx]), self.Filenames[idx]

#     def __len__(self):
#         return len(self.Images)


import glob
import os
from copy import deepcopy

import numpy as np
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class p2_dataset(Dataset):
    def __init__(self, path, transform, train=False) -> None:
        super().__init__()

        self.Train = train
        self.Transform = transform
        self.Image_names = sorted(glob.glob(os.path.join(path, "*.jpg")))

        if self.Train:
            self.Mask_names = sorted(glob.glob(os.path.join(path, "*.png")))

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
                     transform=transforms.ToTensor(), train=True)
    for x, y in dst:
        # print(x, y)
        break
    # mean = torch.zeros(3)
    # std = torch.zeros(3)
    # for x, _ in dst:
    #     mean += x.mean(dim=(1, 2))
    #     std += x.std(dim=(1, 2))

    # mean /= len(dst)
    # std /= len(dst)
    # print(mean, std)
    # tensor([0.4085, 0.3785, 0.2809]) tensor([0.1155, 0.0895, 0.0772])

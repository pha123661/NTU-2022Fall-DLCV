import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import os
import random
import numpy as np

from plyfile import PlyData, PlyElement
import pandas as pd

from util.scannet200 import VALID_CLASS_IDS_200, CLASS_LABELS_200
from util.voxelize import voxelize


def read_plyfile(filepath):
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


class scannet200(Dataset):
    def __init__(self, root, split_dir, transform=None, args=None):
        """ Intialize the dataset """
        split_flie = open(split_dir, "r")
        fn_list = []
        for line in split_flie:
            stripped_line = line.strip()
            fn_list.append(stripped_line.split('/')[-1])
        self.filenames = [os.path.join(root, x) for x in fn_list]
        self.root = root
        self.transform = transform
        self.mix3d = args.mix3d

        self.voxel_size = args.voxel_size
        self.voxel_max = args.voxel_max
        # discard other labels not in the 200 classes
        global g_ignore_idx
        g_ignore_idx = 255
        # self.label_transform = np.vectorize(label_transform)

        self.len = len(self.filenames)
                              
    def __getitem__(self, index):
        """ Get a sample from the dataset """
        ply_fn = self.filenames[index]
        dataframe = read_plyfile(ply_fn)
        coords, colors = dataframe[:, :3], dataframe[:, 3:6]# / 255.

        idx_list, coord_list, feat_list, offset_list = data_prepare(
            coords, colors, self.voxel_size, self.voxel_max, self.transform, self.mix3d)
        return idx_list, coord_list, feat_list, offset_list, ply_fn, coords.shape[0]

    def collate_fn(self, samples):
        return samples[0][0], samples[0][1], samples[0][2], samples[0][3], samples[0][4], samples[0][5]

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


def normalize(coord, feat, mix3d=False):
    if mix3d:
        coord_shift = np.mean(coord, 0)
        coord -= coord_shift
    else:
        coord_shift = np.min(coord, 0)
        coord -= coord_shift
    # coord = torch.FloatTensor(coord)
    # feat = torch.FloatTensor(feat) / 255.
    return coord, feat / 255.


def data_prepare(coord, feat, voxel_size=0.04, voxel_max=None, transform=None, mix3d=False):
    # if transform:
    #     coord, feat = transform(coord, feat)
    if voxel_size:
        coord_min = np.min(coord, 0)
        coord -= coord_min
        idx_sort, count = voxelize(coord, voxel_size, mode=1)

        idx_list, coord_list, feat_list, offset_list  = [], [], [], []
        for i in range(count.max()):
            idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
            idx_part = idx_sort[idx_select]

            coord_part, feat_part = coord[idx_part], feat[idx_part]
            coord_part, feat_part = normalize(coord_part, feat_part, mix3d)

            idx_list.append(idx_part), coord_list.append(coord_part)
            feat_list.append(feat_part), offset_list.append(idx_part.size)
    return idx_list, coord_list, feat_list, offset_list


label2idx = {v: i for i, v in enumerate(VALID_CLASS_IDS_200)}
g_ignore_idx = None


def label_transform(label):
    global g_ignore_idx
    if label in label2idx:
        return label2idx[label]
    else:
        return g_ignore_idx
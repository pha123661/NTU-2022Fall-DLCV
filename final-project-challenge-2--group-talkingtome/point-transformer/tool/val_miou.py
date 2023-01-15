import argparse
import os
from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from plyfile import PlyData

from util.common_util import AverageMeter, intersectionAndUnionGPU
from util.scannet200 import VALID_CLASS_IDS_200

parser = argparse.ArgumentParser(
    description='PyTorch Point Cloud Semantic Segmentation')
parser.add_argument(
    '--sub_root', type=str, default="/tmp3/ycwei/test_inference/2022_fall_DLCV_final/point-transformer/submission")
parser.add_argument(
    '--gt_root', type=str, default="/tmp3/ycwei/2022_fall_DLCV_final/point-transformer/dataset/scannet200")
parser.add_argument(
    '--split_dir', type=str, default="/tmp3/ycwei/2022_fall_DLCV_final/point-transformer/dataset/scannet200/val_split.txt")


def read_plyfile(filepath):
    with open(filepath, 'rb') as f:
        plydata = PlyData.read(f)
    if plydata.elements:
        return pd.DataFrame(plydata.elements[0].data).values


if __name__ == '__main__':
    args = parser.parse_args()

    sub_filenames = [os.path.join(args.sub_root, x) for x in os.listdir(
        args.sub_root) if x.endswith(".txt")]
    sub_filenames.sort()

    split_flie = open(args.split_dir, "r")
    fn_list = []
    for line in split_flie:
        stripped_line = line.strip()
        fn_list.append(stripped_line)
    gt_filenames = [os.path.join(args.gt_root, x) for x in fn_list]
    gt_filenames.sort()

    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    label2idx = {v: i for i, v in enumerate(VALID_CLASS_IDS_200)}

    label_2idx = defaultdict(lambda: -1, label2idx)
    label_transform = np.vectorize(lambda x: label_2idx[x])
    for sub_fn, gt_fn in zip(sub_filenames, gt_filenames):
        # print(sub_fn, gt_fn)
        with open(sub_fn) as f:
            pred = np.asarray([int(x) for x in f.read().split()])
        target = read_plyfile(gt_fn)[:, -2]

        pred = label_transform(pred)
        target = label_transform(target)

        pred = torch.FloatTensor(pred).cuda(non_blocking=True)
        target = torch.FloatTensor(target).cuda(non_blocking=True)

        intersection, union, target = intersectionAndUnionGPU(
            pred, target, 200, -1)
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(
            union), target_meter.update(target)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    print('Val result:')
    print(
        f"mIoU: {iou_class.mean()}, std: {iou_class.std()}, max: {iou_class.max()}, min: {iou_class.min()}")
    print('mAcc : {:.6f}\nallAcc : {:.6f}'.format(mAcc, allAcc))

    sns.set(rc={'figure.figsize': (15, 7.5)})
    sns.set_context('paper')
    ax = sns.barplot(x=np.arange(200), y=-np.sort(-iou_class))
    ax.set(title="Per-Class IoU", xticklabels=[])
    ax.tick_params(bottom=False)
    sns.despine(ax=ax)
    ax.get_figure().savefig('per_class_IoU_trans', dpi=800, transparent=True)

    sns.set_style(style="darkgrid")
    ax.get_figure().savefig('per_class_IoU_white', dpi=800,)

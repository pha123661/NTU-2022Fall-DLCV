import os
import time
import random
import numpy as np
import logging
import pickle
import argparse
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from tqdm import tqdm
from torch.utils.data import DataLoader

from util import config
from util import transform as t
from util.common_util import AverageMeter, intersectionAndUnion, intersectionAndUnionGPU, check_makedirs
from util.voxelize import voxelize


from util.scannet200 import (VALID_CLASS_IDS_200, CLASS_FREQUENCY_200, CLASS_INVERSE_FREQUENCY,
                             SCANNET200)
from tool.dataset import scannet200



def get_parser():
    parser = argparse.ArgumentParser(description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument('--config', type=str, default='config/scannet200/scannet200_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/scannet200/scannet200_pointtransformer_repro.yaml for all options', default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg

def get_logger():
    logger_name = "main-logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def input_normalize(coord, feat):
    coord_min = np.min(coord, 0)
    coord -= coord_min
    feat = feat / 255.
    return coord, feat

def compute_adjustment():
    label_freq = CLASS_FREQUENCY_200
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** args.tro + 1e-12)
    print(args.tro)
    adjustments = torch.from_numpy(adjustments).float()
    adjustments = adjustments.to("cuda")
    return adjustments

if __name__ == '__main__':
    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed(123)
    device = torch.device('cuda' if (torch.cuda.is_available()) else 'cpu')
    print(f"Using {device}")


    global args, logger
    args = get_parser()
    print(args)
    logger = get_logger()
    logger.info(args)
    assert args.classes > 1
    logger.info("=> creating model ...")
    logger.info("Classes: {}".format(args.classes))

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import pointtransformer_seg_repro as Model

    model = Model(c=args.fea_dim, k=args.classes).cuda()
    logger.info(model)
    

    testset = scannet200(root = args.data_dir, 
                            split_dir = args.split_txt_dir,
                            args=args)
    testset_loader = DataLoader(testset, batch_size=1, collate_fn = testset.collate_fn, shuffle=False, num_workers=0)
    print('# images in trainset:', len(testset))


    checkpoint = torch.load(args.model_path)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if args.logit_adj_post:
        args.logit_adjustments = compute_adjustment()

    idx2label = {i: v for i, v in enumerate(VALID_CLASS_IDS_200)}
    #Evaluation loop
    pbar = tqdm(testset_loader, position=0, leave=True)
    with torch.no_grad():
        for idx_list, coord_list, feat_list, offset_list, fn, size in pbar:  # (n, 3), (n, c), (n), (b)
            pred = torch.zeros((size, args.classes)).cuda()
            batch_num = int(np.ceil(len(idx_list) / args.batch_size_test))
            for i in range(batch_num):
                s_i, e_i = i * args.batch_size_test, min((i + 1) * args.batch_size_test, len(idx_list))
                idx_part, coord_part, feat_part, offset_part = idx_list[s_i:e_i], coord_list[s_i:e_i], feat_list[s_i:e_i], offset_list[s_i:e_i]
                idx_part = np.concatenate(idx_part)
                coord_part = torch.FloatTensor(np.concatenate(coord_part)).cuda(non_blocking=True)
                feat_part = torch.FloatTensor(np.concatenate(feat_part)).cuda(non_blocking=True)
                offset_part = torch.IntTensor(np.cumsum(offset_part)).cuda(non_blocking=True)
                with torch.no_grad():
                    pred_part = model([coord_part, feat_part, offset_part])  # (n, k)
                    if args.logit_adj_post:
                        pred_part = pred_part - args.logit_adjustments
                torch.cuda.empty_cache()
                pred[idx_part, :] += pred_part

            pred = pred.max(1)[1].data.cpu().numpy()

            pred = np.vectorize(idx2label.get)(pred)

            fn = os.path.basename(fn).split('.')[0] + ".txt"
            np.savetxt(os.path.join(args.output_dir, fn), pred, fmt="%s")
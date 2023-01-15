import argparse
import logging
import os
import random
import shutil
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.utils.data
from tensorboardX import SummaryWriter

from util import config
from util import transform as t
from util.cb_loss import ClassBalancedLoss
from util.common_util import AverageMeter, intersectionAndUnionGPU
from util.data_util import collate_fn
from util.GradualWarmupScheduler import GradualWarmupScheduler
from util.max_norm import MaxNorm
from util.scannet200 import (CLASS_FREQUENCY_200, CLASS_INVERSE_FREQUENCY,
                             SCANNET200)

def reset_all_weights(model: nn.Module) -> None:
    """
    refs:
        - https://discuss.pytorch.org/t/how-to-re-set-alll-parameters-in-a-network/20819/6
        - https://stackoverflow.com/questions/63627997/reset-parameters-of-a-neural-network-in-pytorch
        - https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    """

    @torch.no_grad()
    def weight_reset(m: nn.Module):
        # - check if the current module has reset_parameters & if it's callabed called it on m
        reset_parameters = getattr(m, "reset_parameters", None)
        if callable(reset_parameters):
            print(f"reset weight for {m}")
            m.reset_parameters()

    # Applies fn recursively to every submodule see: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
    model.apply(fn=weight_reset)

def get_parser():
    parser = argparse.ArgumentParser(
        description='PyTorch Point Cloud Semantic Segmentation')
    parser.add_argument(
        '--config', type=str, default='config/s3dis/s3dis_pointtransformer_repro.yaml', help='config file')
    parser.add_argument('opts', help='see config/s3dis/s3dis_pointtransformer_repro.yaml for all options',
                        default=None, nargs=argparse.REMAINDER)
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


def worker_init_fn(worker_id):
    random.seed(args.manual_seed + worker_id)


def main_process():
    return not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % args.ngpus_per_node == 0)


def main():
    args = get_parser()

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)
        torch.cuda.manual_seed(args.manual_seed)
        torch.cuda.manual_seed_all(args.manual_seed)
        cudnn.benchmark = False
        cudnn.deterministic = True

    args.sync_bn = False
    args.distributed = False
    args.multiprocessing_distributed = False

    main_worker(args)


class LogCoshDiceLoss(nn.Module):
    """
    L_{lc-dce} = log(cosh(DiceLoss)
    """

    def __init__(self, use_softmax=True, ignore_index=-1):
        super(LogCoshDiceLoss, self).__init__()
        self.use_softmax = use_softmax
        self.ignore_index = ignore_index

    def forward(self, logits, targets, epsilon=1e-6):
        not_ignore_mask = targets != self.ignore_index
        if len(not_ignore_mask) == 0:
            print('warning: ignoring all elements')
            return torch.tensor(0.0, requires_grad=True)
        # ignore
        logits, targets = logits[not_ignore_mask], targets[not_ignore_mask]
        num_classes = logits.shape[1]
        # Apply softmax to the output to present it in probability.
        if self.use_softmax:
            logits = torch.nn.functional.softmax(logits, dim=1)
        one_hot_target = torch.nn.functional.one_hot(
            targets.to(torch.int64),
            num_classes=num_classes
        ).to(torch.float)
        assert logits.shape == one_hot_target.shape
        # Shape [batch, n_classes]
        numerator = 2. * torch.sum(logits * one_hot_target, dim=(-2, -1))
        denominator = torch.sum(logits + one_hot_target, dim=(-2, -1))
        return torch.log(torch.cosh(1 - torch.mean((numerator + epsilon) / (denominator + epsilon))))


def main_worker(argss):
    global args, best_iou
    args, best_iou = argss, 0

    if args.arch == 'pointtransformer_seg_repro':
        from model.pointtransformer.pointtransformer_seg import \
            pointtransformer_seg_repro as Model
    else:
        raise Exception('architecture not supported yet'.format(args.arch))
    model = Model(c=args.fea_dim, k=args.classes)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if args.criterion == "FocalLoss":
        criterion = torch.hub.load(
            # configure loading model
            'adeelh/pytorch-multi-class-focal-loss',
            model='FocalLoss',
            force_reload=False,

            # focal loss args
            alpha=torch.as_tensor(CLASS_INVERSE_FREQUENCY),
            gamma=2,
            reduction='mean',
            ignore_index=args.ignore_label
        ).cuda()
    elif args.criterion == "CrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
    elif args.criterion == "WeightedCrossEntropyLoss":
        criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label,
                                        weight=torch.as_tensor(CLASS_INVERSE_FREQUENCY)).cuda()
    elif args.criterion == "LogCoshDiceLoss":
        criterion = LogCoshDiceLoss(ignore_index=args.ignore_label)
    elif args.criterion == "ClassBalancedLoss":
        criterion = ClassBalancedLoss(
            samples_per_class=[CLASS_FREQUENCY_200[c] for c in sorted(CLASS_FREQUENCY_200.keys())], ignore_idx=args.ignore_label)
    else:
        if args.criterion is None:
            raise ValueError("Please specify args.criterion in config")
        raise NotImplemented(f"{args.criterion} not supported")

    if args.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(
        ), lr=args.base_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    steps_per_epoch = (1059 * 30) // args.batch_size + \
        (1059 * 30) % args.batch_size
    if args.lr_scheduler == 'OneCycleLR':
        after_scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=args.base_lr, steps_per_epoch=steps_per_epoch,
                                                  epochs=args.epochs - args.warmup_epoch, pct_start=args.lr_pct_start, verbose=True)
    elif args.lr_scheduler == 'MultiStepLR':
        after_scheduler = lr_scheduler.MultiStepLR(
            optimizer, milestones=args.lr_scheduler_milestones, gamma=args.lr_gamma, verbose=True)
    elif args.lr_scheduler == 'CosineAnnealingLR':
        after_scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - args.warmup_epoch, eta_min=0, last_epoch=-1)
    elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
        after_scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=args.T_0, T_mult=args.T_mult, eta_min=args.eta_min, verbose=True)

    # warmup
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_epoch, after_scheduler=after_scheduler)

    if main_process():
        global logger, writer
        logger = get_logger()
        writer = SummaryWriter(args.save_path)
        logger.info(args)
        logger.info("=> creating model ...")
        logger.info("Classes: {}".format(args.classes))
        logger.info(model)
        model = model.cuda()
    else:
        model = torch.nn.DataParallel(model.cuda())

    if args.weight:
        if os.path.isfile(args.weight):
            if main_process():
                logger.info("=> loading weight '{}'".format(args.weight))
            checkpoint = torch.load(args.weight)
            model.load_state_dict(checkpoint['state_dict'])
            if main_process():
                logger.info("=> loaded weight '{}'".format(args.weight))
        else:
            logger.info("=> no weight found at '{}'".format(args.weight))

    if args.resume:
        if os.path.isfile(args.resume):
            if main_process():
                logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage, loc: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            if not (hasattr(args, "max_norm") and args.max_norm is not None):
                optimizer.load_state_dict(checkpoint['optimizer'])
                scheduler.load_state_dict(checkpoint['scheduler'])
            #best_iou = 40.0
            best_iou = checkpoint['best_iou']
            if main_process():
                logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint['epoch']))
        else:
            if main_process():
                logger.info(
                    "=> no checkpoint found at '{}'".format(args.resume))
    if args.mix3d:
        train_transform = t.Compose([
            t.RandomFlip(p=0.2),
            t.RandomRotate(angle=[1 / 128, 1 / 128, 1 / 128]),
            t.RandomScale(scale=[0.9, 1.1]),
            t.ChromaticAutoContrast(p=0.2),
            t.ChromaticTranslation(p=0.95, ratio=0.05),
            t.ChromaticJitter(p=0.95, std=0.005),  # std=0.005
            t.HueSaturationTranslation(),
        ])
    else:
        train_transform = t.Compose([
            t.RandomScale([0.9, 1.1]),
            t.ChromaticAutoContrast(),
            t.ChromaticTranslation(),
            t.ChromaticJitter(),
            t.HueSaturationTranslation()
        ])

    train_data = SCANNET200(split='train', data_root=args.data_root, voxel_size=args.voxel_size, voxel_max=args.voxel_max,
                            transform=train_transform, shuffle_index=True, loop=args.loop, ignore_idx=args.ignore_label, mix3d=args.mix3d)
    if main_process():
        logger.info("train_data samples: '{}'".format(len(train_data)))
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(
        train_sampler is None), num_workers=args.workers, pin_memory=True, sampler=train_sampler, drop_last=True, collate_fn=collate_fn)

    assert steps_per_epoch == len(
        train_loader), f"scheduler is configured with `steps_per_epoch`= {steps_per_epoch}, while training loader has length of {len(train_loader)}"
    val_loader = None
    if args.evaluate:
        val_transform = None
        val_data = SCANNET200(split='val', data_root=args.data_root, voxel_size=args.voxel_size,
                              voxel_max=800000, transform=val_transform, ignore_idx=args.ignore_label)
        if args.distributed:
            val_sampler = torch.utils.data.distributed.DistributedSampler(
                val_data)
        else:
            val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler, collate_fn=collate_fn)

    
    max_norm_applier = None
    if hasattr(args, "max_norm") and args.max_norm is not None:
        assert args.resume, "Must load checkpoint before stage 2 training (max norm + freeze backbone)"
        print('=====> Enable max norm and freeze backbone')
        reset_all_weights(model.cls)
        model = model.cuda()
        max_norm_applier = MaxNorm(args.max_norm)
        # freeze backbone
        for p in model.parameters():
            p.requires_grad = False
        for p in model.cls.parameters():
            p.requires_grad = True
    
    if args.warmup_epoch > 0:
        # warmup scheduler needs to be called once at the start of training
        optimizer.zero_grad()
        optimizer.step()  # to dismiss pytorch warning "scheduler before optimizer"
        scheduler.step()


    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        loss_train, mIoU_train, mAcc_train, allAcc_train = train(
            train_loader, model, criterion, optimizer, epoch, scheduler, max_norm_applier)
        if args.lr_scheduler not in ['OneCycleLR', 'CosineAnnealingWarmRestarts']:
            scheduler.step()
        epoch_log = epoch + 1
        if main_process():
            writer.add_scalar('loss_train', loss_train, epoch_log)
            writer.add_scalar('mIoU_train', mIoU_train, epoch_log)
            writer.add_scalar('mAcc_train', mAcc_train, epoch_log)
            writer.add_scalar('allAcc_train', allAcc_train, epoch_log)

        is_best = False
        if args.evaluate and (epoch_log % args.eval_freq == 0):
            if args.data_name == 'shapenet':
                raise NotImplementedError()
            else:
                loss_val, mIoU_val, mAcc_val, allAcc_val = validate(
                    val_loader, model, criterion)

            if main_process():
                writer.add_scalar('loss_val', loss_val, epoch_log)
                writer.add_scalar('mIoU_val', mIoU_val, epoch_log)
                writer.add_scalar('mAcc_val', mAcc_val, epoch_log)
                writer.add_scalar('allAcc_val', allAcc_val, epoch_log)
                is_best = mIoU_val > best_iou
                best_iou = max(best_iou, mIoU_val)

        if (epoch_log % args.save_freq == 0) and main_process():
            filename = args.save_path + '/model/model_' + str(epoch) + '.pth'
            logger.info('Saving checkpoint to: ' + filename)
            torch.save({'epoch': epoch_log, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(), 'best_iou': best_iou, 'is_best': is_best}, filename)
            if is_best:
                logger.info(
                    'Best validation mIoU updated to: {:.4f}'.format(best_iou))
                shutil.copyfile(filename, args.save_path +
                                '/model/model_best.pth')

    if main_process():
        writer.close()
        logger.info('==>Training done!\nBest Iou: %.3f' % (best_iou))


def compute_adjustment():
    label_freq = CLASS_FREQUENCY_200
    label_freq = dict(sorted(label_freq.items()))
    label_freq_array = np.array(list(label_freq.values()))
    label_freq_array = label_freq_array / label_freq_array.sum()
    adjustments = np.log(label_freq_array ** args.tro_train + 1e-12)
    adjustments = torch.from_numpy(adjustments).float()
    adjustments = adjustments.to("cuda")
    return adjustments


def train(train_loader, model, criterion, optimizer, epoch, scheduler, max_norm_applier=None):

    if args.logit_adj_train:
        args.logit_adjustments = compute_adjustment()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.train()
    end = time.time()
    max_iter = args.epochs * len(train_loader)
    train_loader_len = len(train_loader)
    for i, (coord, feat, target, offset) in enumerate(train_loader):  # (n, 3), (n, c), (n), (b)
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(
            non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        output = model([coord, feat, offset])
        if args.logit_adj_train:
            output = output + args.logit_adjustments
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = criterion(output, target) / args.accumulation_steps
        loss.backward()
        if ((i + 1) % args.accumulation_steps == 0) or (i + 1 == train_loader_len):
            # Update Optimizer
            optimizer.step()
            optimizer.zero_grad()
            if max_norm_applier is not None:
                max_norm_applier(model.cls)

        if args.lr_scheduler == 'OneCycleLR':
            scheduler.step()
        elif args.lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler.step(epoch + i / train_loader_len)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n
        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(
                union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(
            union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / \
            (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()

        # calculate remain time
        current_iter = epoch * len(train_loader) + i + 1
        remain_iter = max_iter - current_iter
        remain_time = remain_iter * batch_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(
            int(t_h), int(t_m), int(t_s))

        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Epoch: [{}/{}][{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Lr {lr:.10f} '
                        'Remain {remain_time} '
                        'Loss {loss_meter.val:.4f} '
                        'mIoU {mIoU:.4f}.'.format(epoch + 1, args.epochs, i + 1, len(train_loader),
                                                  batch_time=batch_time, data_time=data_time,
                                                  lr=optimizer.param_groups[0]['lr'],
                                                  remain_time=remain_time,
                                                  loss_meter=loss_meter,
                                                  mIoU=np.mean(intersection / (union + 1e-10))))
        if main_process():
            writer.add_scalar(
                'learning_rate', optimizer.param_groups[0]['lr'], current_iter)  # log learning rate
            writer.add_scalar('loss_train_batch', loss_meter.val, current_iter)
            writer.add_scalar('mIoU_train_batch', np.mean(
                intersection / (union + 1e-10)), current_iter)
            writer.add_scalar('mAcc_train_batch', np.mean(
                intersection / (target + 1e-10)), current_iter)
            writer.add_scalar('allAcc_train_batch', accuracy, current_iter)

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    if main_process():
        logger.info('Train result at epoch [{}/{}]: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(
            epoch + 1, args.epochs, mIoU, mAcc, allAcc))
    return loss_meter.avg, mIoU, mAcc, allAcc


def validate(val_loader, model, criterion):
    if main_process():
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()

    model.eval()
    end = time.time()
    for i, (coord, feat, target, offset) in enumerate(val_loader):
        data_time.update(time.time() - end)
        coord, feat, target, offset = coord.cuda(non_blocking=True), feat.cuda(
            non_blocking=True), target.cuda(non_blocking=True), offset.cuda(non_blocking=True)
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        with torch.no_grad():
            output = model([coord, feat, offset])
            if args.logit_adj_train:
                # print(output)
                output = output + args.logit_adjustments
                # print(output)
        loss = criterion(output, target)

        output = output.max(1)[1]
        n = coord.size(0)
        if args.multiprocessing_distributed:
            loss *= n
            count = target.new_tensor([n], dtype=torch.long)
            dist.all_reduce(loss), dist.all_reduce(count)
            n = count.item()
            loss /= n

        intersection, union, target = intersectionAndUnionGPU(
            output, target, args.classes, args.ignore_label)
        if args.multiprocessing_distributed:
            dist.all_reduce(intersection), dist.all_reduce(
                union), dist.all_reduce(target)
        intersection, union, target = intersection.cpu(
        ).numpy(), union.cpu().numpy(), target.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(
            union), target_meter.update(target)

        accuracy = sum(intersection_meter.val) / \
            (sum(target_meter.val) + 1e-10)
        loss_meter.update(loss.item(), n)
        batch_time.update(time.time() - end)
        end = time.time()
        if (i + 1) % args.print_freq == 0 and main_process():
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss {loss_meter.val:.4f} ({loss_meter.avg:.4f}) '
                        'Accuracy {accuracy:.4f}.'.format(i + 1, len(val_loader),
                                                          data_time=data_time,
                                                          batch_time=batch_time,
                                                          loss_meter=loss_meter,
                                                          accuracy=accuracy))

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)

    if main_process():
        logger.info(
            'Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(args.classes):
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i,
                        iou_class[i], accuracy_class[i]))
        logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')

    return loss_meter.avg, mIoU, mAcc, allAcc


if __name__ == '__main__':
    import gc
    gc.collect()
    main()

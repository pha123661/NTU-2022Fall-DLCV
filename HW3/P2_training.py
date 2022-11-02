import argparse
import pathlib
import shutil

import torch
from language_evaluation import CocoEvaluator
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm

from ICDataset import ICDataset
from P2_model import ImageCaptioningTransformer
from warmup_scheduler import GradualWarmupScheduler


def main(args):
    # Preprocess
    tokenizer = Tokenizer.from_file(args.tokenizer)
    transform = create_transform(**resolve_data_config({}, model=args.model))
    train_set = ICDataset(
        image_dir=args.train_image_dir,
        json_file=args.train_info,
        transform=transform,
        tokenizer=tokenizer
    )
    valid_set = ICDataset(
        image_dir=args.valid_image_dir,
        json_file=args.valid_info,
        transform=transform,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_set, args.batch_size,
                              collate_fn=train_set.collate_fn,
                              num_workers=6)
    valid_loader = DataLoader(valid_set, args.batch_size,
                              collate_fn=valid_set.collate_fn,
                              num_workers=6)

    Transformer = ImageCaptioningTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        encoder=args.model,
        num_layers=6,
        nhead=8,
        d_model=768,
    )
    Transformer = Transformer.to(args.device)

    # Training
    optimizer = torch.optim.AdamW(Transformer.parameters(), lr=args.lr)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_steps, after_scheduler=cos_scheduler)

    # Log/Validation
    shutil.rmtree(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    log_global_step = 0
    evaluator = CocoEvaluator(coco_types=["CIDEr"], unk_token="[UNK]")

    for epoch in range(args.epochs):
        # Training loop
        for data in tqdm(train_loader):
            scheduler.step()
            optimizer.zero_grad()
            data['images'] = data['images'].to(args.device)
            data['input_ids'] = data['input_ids'].to(args.device)
            loss = Transformer(
                batch_image=data['images'],
                input_ids=data['input_ids']
            )
            loss.backward()
            optimizer.step()

            writer.add_scalar(
                "training/lr", optimizer.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)
            log_global_step += 1

        # Validation loop
        for data in tqdm(valid_loader):
            pass


def parse():
    parser = argparse.ArgumentParser()
    # Path args
    parser.add_argument('--train_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/train')
    parser.add_argument('--valid_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--train_info', type=pathlib.Path,
                        default='hw3_data/p2_data/train.json')
    parser.add_argument('--valid_info', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')
    parser.add_argument("--tensorboard_path",
                        type=pathlib.Path, default="./p2_tb")

    parser.add_argument('--model', type=str, default='vit_base_patch16_224')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')

    # Training args
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args)

import argparse
import json
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
    # Log/Validation
    log_global_step = 0
    history_best = 10e10
    if args.tensorboard_path.exists():
        shutil.rmtree(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    evaluator = CocoEvaluator(coco_types=["CIDEr"], unk_token="[UNK]")
    profiler = torch.profiler.profile(
        schedule=torch.profiler.schedule(
            wait=1, warmup=10, active=500, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(
            str(args.tensorboard_path / 'profiles')),
        record_shapes=True,
        with_stack=True
    )

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
                              shuffle=True,
                              num_workers=6)
    valid_loader = DataLoader(valid_set, 2 * args.batch_size,
                              collate_fn=valid_set.collate_fn,
                              shuffle=False,
                              num_workers=6)

    Transformer = ImageCaptioningTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        encoder=args.model,
        num_layers=4,
        nhead=12,
        d_model=768,
    )
    json.dump(Transformer.config, (args.ckpt_dir /
              f"model_config.json").open(mode='w'), indent=4)
    if torch.cuda.device_count() > 1:
        print(f"## Using {torch.cuda.device_count()} GPUs")
        Transformer = torch.nn.DataParallel(Transformer)
    Transformer = Transformer.to(args.device)
    # Training
    optimizer = torch.optim.AdamW(Transformer.parameters(), lr=args.lr)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader))
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_steps, after_scheduler=cos_scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=not args.disable_fp16)

    # Misc
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    profiler.start()
    for epoch in range(args.epochs):
        # Training loop
        for data in tqdm(train_loader):
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            data['images'] = data['images'].to(args.device, non_blocking=True)
            data['input_ids'] = data['input_ids'].to(
                args.device, non_blocking=True)

            with torch.autocast(device_type='cpu' if args.device == torch.device('cpu') else 'cuda',
                                dtype=torch.float16, enabled=not args.disable_fp16):
                loss = Transformer(
                    batch_image=data['images'],
                    input_ids=data['input_ids']
                )
                loss = loss.sum()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            profiler.step()

            writer.add_scalar(
                "training/lr", optimizer.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)
            log_global_step += 1

        # Validation loop
        for data in tqdm(valid_loader):
            data['images'] = data['images'].to(args.device, non_blocking=True)
            data['input_ids'] = data['input_ids'].to(
                args.device, non_blocking=True)

            va_losses = []
            with torch.no_grad():
                with torch.autocast(device_type='cpu' if args.device == torch.device('cpu') else 'cuda',
                                    dtype=torch.float16, enabled=not args.disable_fp16):
                    loss = Transformer(
                        batch_image=data['images'],
                        input_ids=data['input_ids']
                    )
            va_losses.append(loss.item())

        va_loss = sum(va_losses) / len(va_losses)
        writer.add_scalar("validation/loss", va_loss, global_step=epoch)
        if va_loss < history_best:
            history_best = va_loss
            if isinstance(Transformer, torch.nn.DataParallel):
                print('save module')
                torch.save(Transformer.module.state_dict(),
                           args.ckpt_dir / "Best_model.pth")
            else:
                torch.save(Transformer.state_dict(),
                           args.ckpt_dir / "Best_model.pth")
            print(f'saved model with metric={va_loss}')

    profiler.stop()


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--train_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/train')
    parser.add_argument('--valid_image_dir', type=pathlib.Path,
                        default='hw3_data/p2_data/images/val')
    parser.add_argument('--train_info', type=pathlib.Path,
                        default='hw3_data/p2_data/train.json')
    parser.add_argument('--valid_info', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')
    parser.add_argument('--model', type=str,
                        default='vit_base_patch16_224')
    parser.add_argument('--tokenizer', type=str,
                        default='./hw3_data/caption_tokenizer.json')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output Path
    parser.add_argument("--tensorboard_path",
                        type=pathlib.Path, default="./P2_tb")
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Training args
    parser.add_argument("--disable_fp16", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    main(args)

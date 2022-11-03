import argparse
import json
import pathlib
import shutil

from PIL import Image
import clip
import torch
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm.auto import tqdm
from torchvision import transforms

from ICDataset import ICDataset
from P2_model import ImageCaptioningTransformer
from warmup_scheduler import GradualWarmupScheduler


def main(args):
    # Preprocess
    tokenizer = Tokenizer.from_file(args.tokenizer)
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
    ])
    transform = create_transform(**resolve_data_config({}, model=args.model))
    train_set = ICDataset(
        image_dir=args.train_image_dir,
        json_file=args.train_info,
        transform=transforms.Compose([
            augmentation_transforms,
            transform,
        ]),
        tokenizer=tokenizer
    )
    valid_set = ICDataset(
        image_dir=args.valid_image_dir,
        json_file=args.valid_info,
        transform=transform,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              num_workers=4 * torch.cuda.device_count(),
                              pin_memory=True)
    valid_loader = DataLoader(valid_set,
                              batch_size=2 * args.batch_size,
                              shuffle=False,
                              num_workers=4 * torch.cuda.device_count(),
                              pin_memory=True)
    if 'base' in args.model:
        Model = ImageCaptioningTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            encoder=args.model,
            num_layers=4,
            nhead=12,
            d_model=768,
            dropout=0.1,
        )
    elif 'large' in args.model:
        Model = ImageCaptioningTransformer(
            vocab_size=tokenizer.get_vocab_size(),
            encoder=args.model,
            num_layers=12,
            nhead=16,
            d_model=1024,
            dropout=0.1,
        )
    else:
        raise Exception(f"Cannot auto config {args.model}")
    json.dump(Model.config, (args.ckpt_dir /
              f"model_config.json").open(mode='w'), indent=4)

    if torch.cuda.device_count() > 1:
        Model = torch.nn.DataParallel(Model)
    Model = Model.to(args.device)
    print(f"## Using {torch.cuda.device_count()} GPUs")

    # Training
    amp_enable = any([args.fp16, args.bf16])
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(
            f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")

    optimizer = torch.optim.AdamW(Model.parameters(), lr=args.lr)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader) - args.warmup_steps)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_steps, after_scheduler=cos_scheduler)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    # Log/Validation
    log_global_step = 0
    history_best = 0
    if args.tensorboard_path.exists():
        shutil.rmtree(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    model, image_process = clip.load("ViT-B/32", device=args.device)

    # Misc
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()

    for epoch in range(args.epochs):
        # Training loop
        pbar = tqdm(train_loader)
        for data in pbar:
            # Prepare data
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            data['images'] = data['images'].to(args.device, non_blocking=True)
            data['input_ids'] = data['input_ids'].to(
                args.device, non_blocking=True)

            # Get loss
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                loss = Model(
                    batch_image=data['images'],
                    input_ids=data['input_ids']
                )
                loss = loss.sum()

            # Update
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                Model.parameters(), 1.0).detach().item()
            scaler.step(optimizer)
            scaler.update()

            # Log
            writer.add_scalar("training/lr",
                              optimizer.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/gradient norm",
                              grad_norm, global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)
            pbar.set_description(f"Loss={loss.item():.2f}")

            log_global_step += 1

        # Validation loop
        clip_scores = []
        Model.eval()
        pbar = tqdm(enumerate(valid_set), total=1000)
        pbar.set_description(f"Best metric={history_best}")
        for cnt, data in pbar:
            if cnt >= 1000:
                break
            # Generate sentence
            with torch.no_grad():
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                    if torch.cuda.device_count() > 1:
                        output_ids = Model.module.greedy_search(
                            data['image'].to(args.device))
                    else:
                        output_ids = Model.greedy_search(
                            data['image'].to(args.device))
            gen_sentence = tokenizer.decode(output_ids)

            # Preprocess clip features
            clip_image = Image.open(
                args.valid_image_dir / f"{data['filename']}.jpg").convert('RGB')
            clip_image = image_process(
                clip_image).unsqueeze(0).to(args.device)
            text = clip.tokenize(gen_sentence).to(args.device)

            # Calculate similarity
            with torch.no_grad():
                image_features = model.encode_image(clip_image)
                text_features = model.encode_text(text)
            image_features /= image_features.norm(
                dim=-1, keepdim=True)
            text_features /= text_features.norm(
                dim=-1, keepdim=True)
            sim = image_features @ text_features.T
            score = 2.5 * max(sim.item(), 0)
            clip_scores.append(score)
        Model.train()
        clip_score = sum(clip_scores) / len(clip_scores)

        writer.add_scalar("validation/CLIPscore",
                          clip_score, global_step=epoch)


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
                        default='beitv2_large_patch16_224_in22k')  #
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
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    main(args)

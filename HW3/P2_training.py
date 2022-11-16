import argparse
import json
import os
import pathlib
import shutil
from collections import defaultdict

import clip
import language_evaluation
import matplotlib.pyplot as plt
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from tokenizers import Tokenizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm.auto import tqdm

from ICDataset import ICDataset, Image_dataset
from P2_model import ImageCaptioningTransformer
from warmup_scheduler import GradualWarmupScheduler


class ScheduledSampling:
    def __init__(self, total_steps, enabled=True) -> None:
        self.total_steps = total_steps
        self.current_steps = 0

    def step(self) -> float:
        if not self.enabled or self.current_steps >= self.total_steps:
            return 0
        ratio = (self.total_steps - self.current_steps) / self.total_steps
        self.current_steps += 1
        return ratio


def main(args):
    # Preprocess
    tokenizer = Tokenizer.from_file(args.tokenizer)
    augmentation_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(),
    ])
    transform_config = resolve_data_config(
        {},
        model=timm.create_model(args.model, pretrained=True, num_classes=0)
    )
    transform = create_transform(**transform_config)
    train_set = ICDataset(
        image_dir=args.train_image_dir,
        json_file=args.train_info,
        transform=transforms.Compose([
            augmentation_transforms,
            transform,
        ]),
        tokenizer=tokenizer,
    )
    valid_set = Image_dataset(
        root=args.valid_image_dir,
        transform=transform,
    )

    train_loader = DataLoader(train_set,
                              batch_size=args.batch_size,
                              collate_fn=train_set.collate_fn,
                              shuffle=True,
                              num_workers=4,
                              pin_memory=True)

    if 'base' in args.model:
        d_model = 768
        nhead = 8
    elif 'large' in args.model:
        d_model = 1024
        nhead = 8
    else:
        raise Exception(f"Cannot auto config {args.model}")
    if args.nhead is not None:
        nhead = args.nhead
    model = ImageCaptioningTransformer(
        vocab_size=tokenizer.get_vocab_size(),
        encoder=args.model,
        num_layers=args.num_layers,
        nhead=nhead,
        d_model=d_model,
        dropout=0.1,
    )
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    print(
        f"## Model #param={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6}M")
    model = model.to(args.device)

    # Training
    amp_enable = any([args.fp16, args.bf16])
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(
            f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    cos_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs * len(train_loader) - args.warmup_steps)
    scheduler = GradualWarmupScheduler(
        optimizer, multiplier=1, total_epoch=args.warmup_steps, after_scheduler=cos_scheduler)
    sample_scheduler = ScheduledSampling(
        total_steps=args.epochs * len(train_loader), enabled=args.scheduled_sampling)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)

    # Log/Validation
    log_global_step = 0
    history_best_CLIPscore = 0
    history_best_CIDEr = 0
    if args.tensorboard_path.exists():
        shutil.rmtree(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)

    # clip
    clip_model, image_process = clip.load("ViT-B/32", device=args.device)

    # Misc
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()

    # CIDEr
    evaluator = language_evaluation.CocoEvaluator(coco_types=['CIDEr'])
    info = json.load(args.valid_info.open(mode='r'))
    annotations = defaultdict(list)
    for data in info['annotations']:
        annotations[data['image_id']].append(data['caption'])
    img2id = {os.path.splitext(data['file_name'])[0]: data['id']
              for data in info['images']}

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
                loss = model(
                    batch_image=data['images'],
                    input_ids=data['input_ids'],
                    sampleing_ratio=sample_scheduler.step(),
                )

            # Update
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0).detach().item()
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

        preds = dict()
        model.eval()
        # Validation loop
        clip_scores = []
        for cnt, (img, name) in tqdm(enumerate(valid_set), total=len(valid_set)):

            # Generate sentence
            with torch.no_grad():
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                    output_ids = model.greedy_search(
                        img.to(args.device))
            gen_sentence = tokenizer.decode(output_ids)
            preds[name] = gen_sentence

            # Preprocess clip features
            raw_image = Image.open(
                args.valid_image_dir / f"{name}.jpg").convert('RGB')
            clip_image = image_process(
                raw_image).unsqueeze(0).to(args.device)
            text = clip.tokenize(gen_sentence).to(args.device)

            if cnt == 0:
                fig, axes = plt.subplots(3, 1, figsize=(15, 15))
            if cnt < 3:
                axes[cnt].imshow(raw_image)
                axes[cnt].set_title(gen_sentence)
            if cnt == 2:
                writer.add_figure('validation/samples',
                                  fig, global_step=epoch)

            # Calculate similarity
            with torch.no_grad():
                image_features = clip_model.encode_image(clip_image)
                text_features = clip_model.encode_text(text)
            image_features /= image_features.norm(
                dim=-1, keepdim=True)
            text_features /= text_features.norm(
                dim=-1, keepdim=True)
            sim = image_features @ text_features.T
            score = 2.5 * max(sim.item(), 0)
            clip_scores.append(score)
        clip_score = sum(clip_scores) / len(clip_scores)
        print(f'epoch {epoch}: CLIP score={clip_score}')
        writer.add_scalar("validation/CLIPscore",
                          clip_score, global_step=epoch)
        if clip_score > history_best_CLIPscore:
            history_best_CLIPscore = clip_score
            torch.save(model.state_dict(),
                       args.ckpt_dir / "CLIPscore" / "Best_model.pth")
            json.dump(model.config, (args.ckpt_dir / "CLIPscore" /
                                     f"model_config.json").open(mode='w'), indent=4)
            print(f'## Saved model with CLIPs={clip_score}')

        all_preds = []
        all_ans = []
        for image_name, text in preds.items():
            all_ans.append(annotations[img2id[image_name]])
            all_preds.append(text)
        CIDEr_score = evaluator.run_evaluation(
            all_preds, all_ans)['CIDEr']
        print(f'epoch {epoch}: CIDEr score={CIDEr_score}')
        writer.add_scalar("validation/CIDEr",
                          CIDEr_score, global_step=epoch)

        if CIDEr_score > history_best_CIDEr:
            history_best_CIDEr = CIDEr_score
            torch.save(model.state_dict(),
                       args.ckpt_dir / "CIDEr" / "Best_model.pth")
            json.dump(model.config, (args.ckpt_dir / "CIDEr" /
                                     f"model_config.json").open(mode='w'), indent=4)
            print(f'## Saved model with CIDEr={CIDEr_score}')

        model.train()


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
    parser.add_argument("--scheduled_sampling", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=64)

    # Model
    parser.add_argument('--model', type=str,
                        default='deit3_large_patch16_224_in21ft1k')
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--nhead", type=int, default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    (args.ckpt_dir / "CLIPscore").mkdir(exist_ok=True, parents=True)
    (args.ckpt_dir / "CIDEr").mkdir(exist_ok=True, parents=True)
    main(args)

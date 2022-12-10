import argparse
import json
import pathlib
import shutil

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from P2_dataloader import ImageFolderDataset
from P2_model import Classifier, RandomApply
from warmup_scheduler import GradualWarmupScheduler


def main(args):
    image_size = 128
    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        RandomApply(
            transforms.ColorJitter(0.8, 0.8, 0.8, 0.2),
            p=0.3
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomHorizontalFlip(),
        RandomApply(
            transforms.GaussianBlur((3, 3), (1.0, 2.0)),
            p=0.2
        ),
        transforms.RandomResizedCrop((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=torch.tensor([0.485, 0.456, 0.406]),
            std=torch.tensor([0.229, 0.224, 0.225])
        ),
    ])
    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_set = ImageFolderDataset(
        args.train_image_dir,
        transform=train_transform,
        label_csv=args.train_label,
    )
    val_set = ImageFolderDataset(
        args.val_image_dir,
        transform=val_transform,
        label_csv=args.val_label,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=12, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size * 2, shuffle=False, num_workers=4, pin_memory=True
    )

    backbone = models.resnet50(weights=None)
    if args.backbone is None:
        print('No backbone provided, initialize backbone from scratch')
    else:
        backbone.load_state_dict(torch.load(
            args.backbone, map_location=args.device))
        print(f'Loaded backbone from {args.backbone}')

    backbone = backbone.to(args.device)
    if args.freeze_backbone:
        for p in backbone.parameters():
            p.requires_grad = False
        print('Freezed backbone')

    model = Classifier(
        backbone=backbone,
        in_features=1000,
        n_class=len(train_set.label2idx),
        n_layers=args.n_layers,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
    ).to(args.device)

    # Training
    amp_enable = any([args.fp16, args.bf16])
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(
            f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")

    criterion = torch.nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enable)
    scheduler = GradualWarmupScheduler(
        optim,
        multiplier=1,
        total_epoch=args.warmup_steps,
        after_scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
            optim,
            T_max=args.epochs * len(train_loader) - args.warmup_steps
        )
    )

    # Logging
    if args.tb_dir.exists():
        shutil.rmtree(args.tb_dir)
    writer = SummaryWriter(args.tb_dir)
    log_global_step = 0
    history_best = 0
    optim.zero_grad(set_to_none=True)
    optim.step()

    training_args_dict = {k: str(v) for k, v in vars(args).items()}
    print(f"## Training args: {json.dumps(training_args_dict, indent=2)}")

    for epoch in range(args.epochs):
        # Training
        for data in tqdm(train_loader):
            # Forward & Backpropagation
            img = data['img'].to(args.device)
            label = data['label'].to(args.device)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                logits = model(img)
                loss = criterion(logits, label)

            # Update
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()

            # Log
            writer.add_scalar("training/lr",
                              optim.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)

            log_global_step += 1

        # Validation
        model.eval()
        va_loss = []
        va_acc = []
        for data in val_loader:
            img = data['img'].to(args.device)
            label = data['label'].to(args.device)
            with torch.no_grad():
                with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                    logits = model(img)
                    va_loss.append(torch.nn.functional.cross_entropy(
                        logits, label).item())
            y_pred = torch.argmax(logits, dim=1)
            va_acc.append(torch.mean(
                (y_pred == label).type(torch.float)).item())

        va_loss = sum(va_loss) / len(va_loss)
        va_acc = sum(va_acc) / len(va_acc)
        writer.add_scalar("validation/loss", va_loss,
                          global_step=log_global_step)
        writer.add_scalar("validation/accuracy", va_acc,
                          global_step=log_global_step)
        model.train()

        print(f"epoch {epoch}: va acc = {va_acc}")
        if va_acc > history_best:
            # Save
            history_best = va_acc
            torch.save(model.state_dict(), args.ckpt_dir /
                       f"best_classifier.pth")
            print('## Saved model')

    print(f"## Training args: {json.dumps(training_args_dict, indent=2)}")
    print(f"## Final Result, va acc = {history_best}")


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--train_image_dir', type=pathlib.Path,
                        default='hw4_data/office/train')
    parser.add_argument('--train_label', type=pathlib.Path,
                        default='hw4_data/office/train.csv')
    parser.add_argument('--val_image_dir', type=pathlib.Path,
                        default='hw4_data/office/val')
    parser.add_argument('--val_label', type=pathlib.Path,
                        default='hw4_data/office/val.csv')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    parser.add_argument('--backbone', type=pathlib.Path)

    # Output Path
    parser.add_argument("--tb_dir",
                        type=pathlib.Path, default="./P2_tb/finetune")
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")
    parser.add_argument('--exp_name', type=str)

    # Training args
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=100)

    # Model args
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    if args.exp_name is not None:
        args.ckpt_dir = args.ckpt_dir / args.exp_name
        args.tb_dir = args.tb_dir / args.exp_name
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    main(args)

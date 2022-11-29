import argparse
import pathlib

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from byol_pytorch import BYOL
from torchvision import models, transforms
import shutil
from tqdm import tqdm

from P2_dataloader import ImageFolderDataset
from warmup_scheduler import GradualWarmupScheduler


def main(args):
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4706, 0.4495, 0.4031],
            std=[0.2176, 0.2152, 0.2150]
        )
    ])

    train_set = ImageFolderDataset(
        args.train_image_dir,
        transform=train_transform,
        # label_csv="hw4_data/mini/train.csv",
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4
    )

    resnet = models.resnet50(pretrained=False)
    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool'
    ).to(args.device)
    # Training
    amp_enable = any([args.fp16, args.bf16])
    amp_dtype = torch.bfloat16 if args.bf16 else torch.float16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(
            f"Enable AMP training using dtype={amp_dtype} on {amp_device_type}")

    optim = torch.optim.Adam(learner.parameters(), lr=args.lr)
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
    if args.tensorboard_path.exists():
        shutil.rmtree(args.tensorboard_path)
    writer = SummaryWriter(args.tensorboard_path)
    log_global_step = 0
    optim.zero_grad(set_to_none=True)
    optim.step()

    for epoch in range(args.epochs):
        for data in tqdm(train_loader):
            # Forward & Backpropagation
            img = data['img'].to(args.device)
            with torch.autocast(device_type=amp_device_type, dtype=amp_dtype, enabled=amp_enable):
                loss = learner(img)

            # Update
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            learner.update_moving_average()  # update moving average of target encoder

            # Log
            writer.add_scalar("training/lr",
                              optim.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)

            log_global_step += 1

        # Save
        torch.save(resnet.state_dict(), args.ckpt_dir / f"{epoch}_net.pth")


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--train_image_dir', type=pathlib.Path,
                        default='hw4_data/mini/train')
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
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    main(args)

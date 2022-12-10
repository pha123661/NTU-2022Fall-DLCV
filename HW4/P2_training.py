import argparse
import pathlib
import shutil
from heapq import heappop, heappush

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import models, transforms
from tqdm import tqdm

from byol_pytorch import BYOL
from P2_dataloader import ImageFolderDataset
from warmup_scheduler import GradualWarmupScheduler


def main(args):
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        # already normalized in BYOL pipeline
        # transforms.Normalize(
        #     mean=[0.485, 0.456, 0.406],
        #     std=[0.229, 0.224, 0.225]
        # )
    ])

    dataset = ImageFolderDataset(
        args.train_image_dir,
        transform=train_transform,
    )
    train_set, valid_set = random_split(dataset, [0.9, 0.1])
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True
    )
    valid_loader = DataLoader(
        valid_set, batch_size=2 * args.batch_size, shuffle=False, num_workers=4, pin_memory=True
    )

    resnet = models.resnet50(weights=None)
    learner = BYOL(
        resnet,
        image_size=128,
        hidden_layer='avgpool',
        use_momentum=False,
    ).to(args.device)
    # Training
    amp_enable = args.fp16
    amp_device_type = 'cpu' if args.device == torch.device('cpu') else 'cuda'
    if amp_enable:
        print(
            f"Enable AMP training on {args.device}:{amp_device_type}")

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
    optim.zero_grad(set_to_none=True)
    optim.step()
    log_global_step = 0
    saved_files = []  # max_heap
    for epoch in range(args.epochs):
        writer.add_scalar('training/epoch', epoch, global_step=log_global_step)
        # Training
        for data in tqdm(train_loader):
            # Forward & Backpropagation
            img = data['img'].to(args.device)
            with torch.autocast(device_type=amp_device_type, enabled=amp_enable):
                loss = learner(img)

            # Update
            scheduler.step()
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            # learner.update_moving_average()  # update moving average of target encoder

            # Log
            writer.add_scalar("training/lr",
                              optim.param_groups[0]['lr'], global_step=log_global_step)
            writer.add_scalar("training/loss", loss.item(),
                              global_step=log_global_step)

            log_global_step += 1

        # Validation
        va_loss = []
        for data in valid_loader:
            # Forward & Backpropagation
            img = data['img'].to(args.device)
            with torch.no_grad():
                with torch.autocast(device_type=amp_device_type, enabled=amp_enable):
                    loss = learner(img)

            # Update
            loss = loss.item()
            va_loss.append(loss)
            # Log
            writer.add_scalar("validation/loss", loss,
                              global_step=log_global_step)

        # Save
        va_loss = sum(va_loss) / len(va_loss)
        print(f"Epoch {epoch}, validation loss: {va_loss}")
        if not saved_files or va_loss < -saved_files[0][0]:
            save_path = args.ckpt_dir / f"{epoch}_backbone.pth"
            torch.save(resnet.state_dict(), save_path)
            print("Saved model")
            while len(saved_files) > args.save_best_k - 1:
                _, popped_state_dict = heappop(saved_files)
                popped_state_dict.unlink()
            heappush(saved_files, (-va_loss, save_path))

    torch.save(resnet.state_dict(), args.ckpt_dir / f"last_backbone.pth")
    print(
        f'Done, best validation loss: {min(saved_files, key=lambda x: -x[0])}')


def parse_args():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--train_image_dir', type=pathlib.Path,
                        default='hw4_data/mini/train')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    # Output Path
    parser.add_argument("--tensorboard_path",
                        type=pathlib.Path, default="./P2_tb/backbone/")
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")
    parser.add_argument('--save_best_k', type=int, default=4)

    # Training args
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--warmup_steps", type=int, default=1000)

    args = parser.parse_args()
    assert args.save_best_k > 0, "--save_best_k must > 0"
    args.ckpt_dir.mkdir(exist_ok=True, parents=True)
    return args


if __name__ == '__main__':
    args = parse_args()
    main(args)

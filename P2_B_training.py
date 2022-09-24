import os

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as trns
from sklearn.metrics import jaccard_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from P2_dataloader import p2_dataset
from UNETpp import NestedUNet

# load data
mean = [0.4085, 0.3785, 0.2809]  # calculated on training set at dataloader.py
std = [0.1155, 0.0895, 0.0772]

train_dataset = p2_dataset(
    'hw1_data/hw1_data/p2_data/train',
    transform=trns.Compose([
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ]),
    train=True,
)

valid_dataset = p2_dataset(
    'hw1_data/hw1_data/p2_data/validation',
    transform=trns.Compose([
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ]),
    train=True,
)

batch_size = 4
train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda')
epochs = 300
best_loss = 5.0
ckpt_path = f'./P2_B_checkpoint'
accum_steps = 3


net = NestedUNet(num_classes=7, deep_supervision=True)
net = net.to(device)
net.train()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=0.001)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)


for epoch in range(1, epochs + 1):
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        outs = net(x)
        loss = 0
        for logits in outs:
            loss += loss_fn(logits, y)
        loss /= accum_steps
        loss /= len(outs)
        loss.backward()
        if (epoch % accum_steps) == 0 or (epoch == epochs):
            optim.step()
            optim.zero_grad()

    if (epoch % accum_steps) == 0 or (epoch == epochs):
        net.eval()
        with torch.no_grad():
            va_loss = 0
            mIOUs = []
            for x, y in tqdm(valid_loader):
                x, y = x.to(device), y.to(device)
                out = net(x)
                pred = out.argmax(dim=1)
                va_loss += nn.functional.cross_entropy(out, y).item()

                pred = pred.detach().cpu().numpy().astype(np.int64)
                y = y.detach().cpu().numpy().astype(np.int64)
                for p, gt in zip(pred, y):
                    mIOUs.append(jaccard_score(
                        gt.flatten(), p.flatten(), average='macro'))

            va_loss /= len(valid_loader)
            mIOU = sum(mIOUs) / len(mIOUs)
        net.train()
        print(f"epoch {epoch}, mIOU = {mIOU}, va_loss = {va_loss}")
        if va_loss <= best_loss:
            best_loss = va_loss
            torch.save(optim.state_dict(), os.path.join(
                ckpt_path, 'best_optimizer.pth'))
            torch.save(net.state_dict(), os.path.join(
                ckpt_path, 'best_model.pth'))
            print("new model saved sucessfully!")

    if (epoch % 10) == 0 or epoch == 1:
        torch.save(net.state_dict(), os.path.join(
            ckpt_path, f'{epoch}_model.pth'))

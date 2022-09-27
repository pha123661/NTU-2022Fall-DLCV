import os

import numpy as np
import torch
import torchvision.transforms as trns
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from P2_dataloader import p2_dataset
from P2_models import createDeepLabv3


def mean_iou_score(pred, labels):
    '''
    Compute mean IoU score over 6 classes
    '''
    mean_ious = []
    for i in range(6):
        tp_fp = np.sum(pred == i)
        tp_fn = np.sum(labels == i)
        tp = np.sum((pred == i) * (labels == i))
        if (tp_fp + tp_fn - tp) == 0:
            continue
        iou = tp / (tp_fp + tp_fn - tp)
        mean_ious.append(iou)

    return sum(mean_ious) / len(mean_ious)


# load data
mean = [0.485, 0.456, 0.406]  # imagenet
std = [0.229, 0.224, 0.225]

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

batch_size = 24

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

device = torch.device('cuda')
epochs = 300
best_loss = 5.0
ckpt_path = f'./P2_B_checkpoint'

# model
net = createDeepLabv3(7)
net = net.to(device)
net.train()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.001, weight_decay=1e-5)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

for epoch in range(1, epochs + 1):
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optim.zero_grad()
        out = net(x)  # no need to calculate soft-max
        logits, aux_logits = out['out'], out['aux']
        loss = loss_fn(logits, y) + 0.1 * loss_fn(aux_logits, y)
        loss.backward()
        optim.step()

    net.eval()
    with torch.no_grad():
        va_loss = 0
        ACCs = []
        for x, y in tqdm(valid_loader):
            x, y = x.to(device), y.to(device)
            out = net(x)['out']
            pred = out.argmax(dim=1)
            va_loss += nn.functional.cross_entropy(out, y).item()

            pred = pred.detach().cpu().numpy().astype(np.int64)
            y = y.detach().cpu().numpy().astype(np.int64)
            ACCs.append(mean_iou_score(pred, y))

        va_loss /= len(valid_loader)
        acc = sum(ACCs) / len(ACCs)
    net.train()

    print(f"epoch {epoch}, Acc = {acc}, va_loss = {va_loss}")
    if va_loss <= best_loss:
        best_loss = va_loss
        torch.save(optim.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer.pth'))
        torch.save(net.state_dict(), os.path.join(ckpt_path, 'best_model.pth'))
        print("new model saved sucessfully!")

    torch.save(net.state_dict(), os.path.join(
        ckpt_path, f'{epoch}_model.pth'))

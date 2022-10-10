import os

import numpy as np
import torch
import torchvision.transforms as trns
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from P2_dataloader import p2_dataset
from P2_models import FCN32s

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

batch_size = 8

train_loader = DataLoader(
    dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

device = torch.device('cuda')
epochs = 100
best_loss = 5.0
ckpt_path = f'./P2_A_checkpoint'

# model
net = FCN32s()
net.copy_params_from_vgg16(models.vgg16(pretrained=True))
net = net.to(device)
net.train()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.SGD(net.parameters(), lr=0.003, momentum=0.9)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

for epoch in range(1, epochs + 1):
    for x, y in tqdm(train_loader):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

        optim.zero_grad()
        logits = net(x)  # no need to calculate soft-max
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

    net.eval()
    with torch.no_grad():
        va_loss = 0
        ACCs = []
        for x, y in tqdm(valid_loader):
            x, y = x.to(device), y.to(device)
            out = net(x)
            pred = out.argmax(dim=1)
            va_loss += nn.functional.cross_entropy(out, y).item()

            pred = pred.detach().cpu().numpy().astype(np.int64)
            y = y.detach().cpu().numpy().astype(np.int64)
            ACCs.append(np.sum(pred == y) / len(y.flatten()))

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
    if (epoch % 10) == 0 or epoch == 1:
        torch.save(net.state_dict(), os.path.join(
            ckpt_path, f'{epoch}_model.pth'))

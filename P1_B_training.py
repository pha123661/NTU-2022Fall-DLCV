import os
import shutil

import numpy as np
import timm
import torch
import torchvision.transforms as trns
from sklearn.metrics import accuracy_score
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from P1_dataloader import p1_dataset

# load data
mean = [0.5077, 0.4813, 0.4312]  # calculated at dataloader.py
std = [0.2000, 0.1986, 0.2034]
train_dataset = p1_dataset(
    '/shared_home/r11944004/pepper_local_disk/DLCV/hw1-pha123661/hw1_data/hw1_data/p1_data/train_50',
    trns.Compose([
        trns.Resize((256, 256)),
        trns.RandomCrop((224, 224)),  # pre-trained model works on 224*224
        trns.RandomHorizontalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ])
)

valid_dataset = p1_dataset(
    '/shared_home/r11944004/pepper_local_disk/DLCV/hw1-pha123661/hw1_data/hw1_data/p1_data/val_50',
    trns.Compose([
        trns.Resize((224, 224)),
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ])
)


train_loader = DataLoader(
    dataset=train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

model = 'beit_base_patch16_224_in22k'
device = torch.device('cuda')
epochs = 1000
plot_epochs = {100 * i for i in range(1, 11)} | {1}
best_acc = 0.6
ckpt_path = f'./p1_B_checkpoint_{model}'
if os.path.isdir('tb'):
    shutil.rmtree('tb', ignore_errors=True)
writer = SummaryWriter('tb')

# model
net = timm.create_model(model, pretrained=True, num_classes=50)
net = net.to(device)
net.train()
loss_fn = SoftTargetCrossEntropy()
optim = torch.optim.AdamW(net.parameters(), lr=0.001)
mixup = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=0.8,
    prob=0.3,
    switch_prob=0.5,
    mode='batch',
    label_smoothing=0.1,
    num_classes=50,
)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

# train
global_step = 0
for epoch in range(1, epochs + 1):
    for x, y in train_loader:
        global_step += 1
        x, y = x.to(device, non_blocking=True), y.to(
            device, non_blocking=True)
        x, y = mixup(x, y)

        optim.zero_grad()
        logits = net(x)  # no need to calculate soft-max
        loss = loss_fn(logits, y)
        loss.backward()
        optim.step()

    net.eval()
    with torch.no_grad():
        va_loss = 0
        va_acc = 0
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            out = net(x)
            va_loss += nn.functional.cross_entropy(out, y).item()
            y_pred = np.argmax(
                out.detach().cpu().numpy(), axis=1).flatten()
            va_acc += accuracy_score(y_pred,
                                     y.detach().cpu().numpy().flatten())
        va_loss = va_loss / len(valid_loader)
        va_acc /= len(valid_loader)
    net.train()
    writer.add_scalar('validation/acc', va_acc, global_step)
    writer.add_scalar('validation/loss', va_loss, global_step)

    print(f"epoch {epoch}, va_acc = {va_acc}, va_loss = {va_loss}")
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(optim.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer.pth'))
        torch.save(net.state_dict(), os.path.join(ckpt_path, 'best_model.pth'))
        print("new model saved sucessfully!")

print(f"best validation acc: {best_acc}")

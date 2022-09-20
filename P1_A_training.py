import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as trns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader

import P1_models
from P1_dataloader import p1_dataset

# load data
mean = [0.5077, 0.4813, 0.4312]  # calculated at dataloader.py
std = [0.2000, 0.1986, 0.2034]
train_dataset = p1_dataset(
    '/shared_home/r11944004/pepper_local_disk/DLCV/hw1-pha123661/hw1_data/hw1_data/p1_data/train_50',
    trns.Compose([
        trns.Resize((32, 32)),
        trns.RandomCrop((32, 32), padding=4),
        trns.RandomHorizontalFlip(),
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ])
)

valid_dataset = p1_dataset(
    '/shared_home/r11944004/pepper_local_disk/DLCV/hw1-pha123661/hw1_data/hw1_data/p1_data/val_50',
    trns.Compose([
        trns.Resize((32, 32)),
        trns.ToTensor(),
        trns.Normalize(mean=mean, std=std),
    ])
)


train_loader = DataLoader(
    dataset=train_dataset, batch_size=128, shuffle=True, num_workers=4)
valid_loader = DataLoader(
    dataset=valid_dataset, batch_size=256, shuffle=False, num_workers=4)

parser = argparse.ArgumentParser()
parser.add_argument(
    '-m', '--model', help='model', type=str)
args = parser.parse_args()

device = torch.device('cuda')
epochs = 1000
plot_epochs = {100 * i for i in range(1, 11)} | {1}
best_acc = 0.0
ckpt_path = f'./p1_A_checkpoint_{args.model}'

# model
net = getattr(P1_models, args.model)()
net = net.to(device)
net.train()
loss_fn = nn.CrossEntropyLoss()
optim = torch.optim.AdamW(net.parameters(), lr=0.001)

if not os.path.isdir(ckpt_path):
    os.mkdir(ckpt_path)

global_step = 0
for epoch in range(1, epochs + 1):
    for x, y in train_loader:
        global_step += 1
        x, y = x.to(device), y.to(device)

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

    # plot
    if epoch in plot_epochs:
        with torch.no_grad():
            all_x = None
            all_y = None
            for x, y in valid_loader:
                x, y = x.to(device), y.to(device)
                out = net.get_embedding(x)  # calling the second last layer
                if all_x is None:
                    all_x = out.detach().cpu().numpy()
                    all_y = y.detach().cpu().numpy().flatten()
                else:
                    out = out.detach().cpu().numpy()
                    y = y.detach().cpu().numpy().flatten()
                    all_x = np.vstack((all_x, out))
                    all_y = np.concatenate((all_y, y))
        # all_x: (2500, 192, 8, 8), all_y: (2500,)
        all_x = all_x.reshape(all_x.shape[0], -1)
        # plot PCA
        pca = PCA(n_components=2)
        d_x = pca.fit_transform(all_x)
        plt.figure()
        plt.title(f"PCA figure for epoch {epoch}")
        plt.scatter(d_x[:, 0], d_x[:, 1], c=all_y)
        plt.savefig(f"./P1_plots/{args.model}_PCA_{epoch}")

        # plot t-SNE
        tsne = TSNE(n_components=2)
        d_x = tsne.fit_transform(all_x)
        plt.figure()
        plt.title(f"t-SNE figure for epoch {epoch}")
        plt.scatter(d_x[:, 0], d_x[:, 1], c=all_y)
        plt.savefig(f"./P1_plots/{args.model}_TSNE_{epoch}")

    print(f"epoch {epoch}, va_acc = {va_acc}, va_loss = {va_loss}")
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(optim.state_dict(), os.path.join(
            ckpt_path, 'best_optimizer.pth'))
        torch.save(net.state_dict(), os.path.join(ckpt_path, 'best_model.pth'))
        print("new model saved  sucessfully!")

print(f"best validation acc: {best_acc}")

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms

from digit_dataloader import digit_dataset
from P3_SVHN_model import FeatureExtractor as SF
from P3_USPS_model import FeatureExtractor as UF

# load dataset
source_val_set = digit_dataset(
    # [0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083]
    root='hw2_data/digits/mnistm/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4631, 0.4666, 0.4195],
                             [0.1979, 0.1845, 0.2083])
    ]),
    label_csv='hw2_data/digits/mnistm/val.csv'
)
SVHN_val_set = digit_dataset(
    # [0.2570, 0.2570, 0.2570], [0.3372, 0.3372, 0.3372]
    root='hw2_data/digits/svhn/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4413, 0.4458, 0.4715],
                             [0.1169, 0.1206, 0.1042])
    ]),
    label_csv='hw2_data/digits/svhn/val.csv'
)
USPS_val_set = digit_dataset(
    # [0.2570, 0.2570, 0.2570], [0.3372, 0.3372, 0.3372]
    root='hw2_data/digits/usps/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.2573, 0.2573, 0.2573],
                             [0.3373, 0.3373, 0.3373])
    ]),
    label_csv='hw2_data/digits/usps/val.csv'
)
fig_path = Path('P3_plot')
fig_path.mkdir(parents=True, exist_ok=True)
batch_size = 1024
source_val_loader = DataLoader(
    source_val_set, batch_size, shuffle=False, num_workers=6)
SVHN_val_loader = DataLoader(
    SVHN_val_set, batch_size, shuffle=False, num_workers=6)
USPS_val_loader = DataLoader(
    USPS_val_set, batch_size, shuffle=False, num_workers=6)

# SVHN
device = 'cuda'
net = SF().to(device)
net.load_state_dict(torch.load('P3_SVHN_ckpt/best_F.pth', map_location=device))
net.eval()

all_feature = []
all_label = []
all_domains = []
for x, y in source_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.zeros((len(x),), dtype=np.int32))

for x, y in SVHN_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.ones((len(x),), dtype=np.int32))

all_feature = np.concatenate(all_feature, axis=0)
all_label = np.concatenate(all_label, axis=0)
all_domains = np.concatenate(all_domains, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feature = TSNE(
    2, init='pca', learning_rate='auto').fit_transform(all_feature)
scatter = axes[0].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_label, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Digits')
axes[0].set_title("Colored by Different Classes")

scatter = axes[1].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_domains, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'Source', 'Target'], title='Domains')
axes[1].set_title("Colored by Different Domains")
fig.savefig(fig_path / 'SVHN')


# USPS
net = UF().to(device)
net.load_state_dict(torch.load('P3_USPS_ckpt/best_F.pth', map_location=device))

all_feature = []
all_label = []
all_domains = []
for x, y in source_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.zeros((len(x),), dtype=np.int32))

for x, y in USPS_val_loader:
    x = x.to(device)
    with torch.no_grad():
        f = net(x)
    all_feature.append(f.cpu().numpy())
    all_label.append(y.cpu().numpy())
    all_domains.append(np.ones((len(x),), dtype=np.int32))

all_feature = np.concatenate(all_feature, axis=0)
all_label = np.concatenate(all_label, axis=0)
all_domains = np.concatenate(all_domains, axis=0)

fig, axes = plt.subplots(1, 2, figsize=(20, 9))
all_feature = TSNE(
    2, init='pca', learning_rate='auto').fit_transform(all_feature)
scatter = axes[0].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_label, alpha=0.5, s=10
)
axes[0].legend(*scatter.legend_elements(), title='Digits')
axes[0].set_title("Colored by Different Classes")

scatter = axes[1].scatter(
    all_feature[..., 0], all_feature[..., 1],
    c=all_domains, alpha=0.5, s=10
)
axes[1].legend(handles=scatter.legend_elements()[0], labels=[
    'Source', 'Target'], title='Domains')
axes[1].set_title("Colored by Different Domains")
fig.savefig(fig_path / 'USPS')

from itertools import chain
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from tqdm import tqdm

from digit_dataloader import digit_dataset
from P3_SVHN_model import DomainClassifier, FeatureExtractor, LabelPredictor


# https://github.com/NaJaeMin92/pytorch_DANN
def rm_tree(pth: Path):
    if pth.is_dir():
        for child in pth.iterdir():
            if child.is_file():
                child.unlink()
            else:
                rm_tree(child)
        pth.rmdir()


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


source_train_set = digit_dataset(
    # [0.4631, 0.4666, 0.4195], [0.1979, 0.1845, 0.2083]
    root='hw2_data/digits/mnistm/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4631, 0.4666, 0.4195],
                             [0.1979, 0.1845, 0.2083])
    ]),
    label_csv='hw2_data/digits/mnistm/train.csv'
)
target_train_set = digit_dataset(
    # [0.2570, 0.2570, 0.2570], [0.3372, 0.3372, 0.3372]
    root='hw2_data/digits/svhn/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4413, 0.4458, 0.4715],
                             [0.1169, 0.1206, 0.1042])
    ]),
    label_csv='hw2_data/digits/svhn/train.csv'
)
target_val_set = digit_dataset(
    # [0.2570, 0.2570, 0.2570], [0.3372, 0.3372, 0.3372]
    root='hw2_data/digits/svhn/data',
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4413, 0.4458, 0.4715],
                             [0.1169, 0.1206, 0.1042])
    ]),
    label_csv='hw2_data/digits/svhn/val.csv'
)

batch_size = 1024
source_train_loader = DataLoader(
    source_train_set, batch_size, shuffle=True, num_workers=6)
target_train_loader = DataLoader(
    target_train_set, batch_size, shuffle=True, num_workers=6)
target_val_loader = DataLoader(
    target_val_set, 2 * batch_size, shuffle=False, num_workers=6)

target_train_loader = iter(cycle(target_train_loader))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckpt_path = Path('./P3_SVHN_ckpt')
tb_path = Path('./P3_SVHN_tb')

rm_tree(ckpt_path)
rm_tree(tb_path)

ckpt_path.mkdir(exist_ok=True)
tb_path.mkdir(exist_ok=True)

writer = SummaryWriter(tb_path)

num_epochs = 200
lr = 0.003
gamma = 10

F = FeatureExtractor().to(device)
L = LabelPredictor().to(device)
D = DomainClassifier().to(device)

label_loss_fn = nn.CrossEntropyLoss()
domain_loss_fn = nn.BCEWithLogitsLoss()
optim = torch.optim.SGD(
    chain(F.parameters(), L.parameters(), D.parameters()), lr=lr, momentum=0.9)

current_step = 0
total_steps = num_epochs * len(source_train_loader)
best_target_acc = 0.3
for epoch in range(num_epochs):
    for (src_x, src_y), (tgt_x, _) in tqdm(zip(source_train_loader, target_train_loader), total=len(source_train_loader)):
        src_x = src_x.to(device, non_blocking=True)
        src_y = src_y.to(device, non_blocking=True)
        tgt_x = tgt_x.to(device, non_blocking=True)

        # scheduling
        p = current_step / total_steps
        lambda_ = 2.0 / (1.0 + np.exp(-gamma * p)) - 1
        optim.param_groups[0]['lr'] = lr / (1.0 + gamma * p) ** 0.75

        # feature extraction
        source_feature = F(src_x)
        target_feature = F(tgt_x)

        # label classification loss
        source_logits = L(source_feature)
        label_loss = label_loss_fn(source_logits, src_y)

        # domain discriminator loss
        # source=1
        source_domain_logit = D(source_feature, lambda_).squeeze()
        source_domain_loss = domain_loss_fn(source_domain_logit, torch.zeros(
            src_x.shape[0], dtype=torch.float, device=device))

        # target=1
        target_domain_logit = D(target_feature, lambda_).squeeze()
        target_domain_loss = domain_loss_fn(target_domain_logit, torch.ones(
            tgt_x.shape[0], dtype=torch.float, device=device))

        domain_loss = source_domain_loss + target_domain_loss

        writer.add_scalars('training', {
                           'label_loss': label_loss, 'domain_loss': domain_loss}, global_step=current_step)

        loss = label_loss + domain_loss
        loss.backward()
        optim.step()
        optim.zero_grad()

        current_step += 1

    # validation
    for model in [F, L, D]:
        model.eval()
    va_acc = 0
    for tgt_x, tgt_y in tqdm(target_val_loader):
        tgt_x = tgt_x.to(device)
        tgt_y = tgt_y.cpu().numpy()

        with torch.no_grad():
            logits = L(F(tgt_x))
        pred = logits.argmax(-1).cpu().numpy()
        va_acc += np.mean((pred == tgt_y).astype(int))
    for model in [F, L, D]:
        model.train()
    va_acc /= len(target_val_loader)
    writer.add_scalar('accuracy/validation', va_acc, global_step=current_step)

    print(f"epoch: {epoch}, va_acc: {va_acc}")
    if va_acc >= best_target_acc:
        best_target_acc = va_acc
        torch.save(F.state_dict(), ckpt_path / f'best_F.pth')
        torch.save(L.state_dict(), ckpt_path / f'best_L.pth')
        torch.save(D.state_dict(), ckpt_path / f'best_D.pth')
        print(f"[new model saved]")

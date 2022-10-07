import os
import subprocess
import sys

import imageio
import numpy as np
import torch
from torchvision import transforms

from P2_dataloader import p2_dataset
from P2_models import DeepLabv3


def pred2image(batch_preds, batch_names, out_path):
    # batch_preds = (b, H, W)
    for pred, name in zip(batch_preds, batch_names):
        pred = pred.detach().cpu().numpy()
        pred_img = np.zeros((512, 512, 3), dtype=np.uint8)
        pred_img[np.where(pred == 0)] = [0, 255, 255]
        pred_img[np.where(pred == 1)] = [255, 255, 0]
        pred_img[np.where(pred == 2)] = [255, 0, 255]
        pred_img[np.where(pred == 3)] = [0, 255, 0]
        pred_img[np.where(pred == 4)] = [0, 0, 255]
        pred_img[np.where(pred == 5)] = [255, 255, 255]
        pred_img[np.where(pred == 6)] = [0, 0, 0]
        imageio.imwrite(os.path.join(
            out_path, name.replace('.jpg', '.png')), pred_img)


device = torch.device(
    'cuda') if torch.cuda.is_available() else torch.device('cpu')


def get_model(path):
    net = DeepLabv3(n_classes=7, mode='resnet')
    net.load_state_dict(torch.load(path))
    net = net.to(device)
    net.eval()
    return net


net_first = get_model('P2_B_checkpoint/1_model.pth')
net_mid = get_model('P2_B_checkpoint/50_model.pth')
net_last = get_model('P2_B_checkpoint/100_model.pth')

input_folder = './p2 plot fig/'
output_folders = ['./p2 plot fig/first/',
                  './p2 plot fig/mid/', './p2 plot fig/last/']

mean = [0.485, 0.456, 0.406]  # imagenet
std = [0.229, 0.224, 0.225]

test_dataset = p2_dataset(
    input_folder,
    transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]),
    train=False
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=6)

for net, output_folder in zip([net_first, net_mid, net_last], output_folders):
    try:
        os.makedirs(output_folder, exist_ok=True)
    except:
        pass
    for x, filenames in test_loader:
        with torch.no_grad():
            x = x.to(device)
            out = net(x)['out']
        pred = out.argmax(dim=1)
        pred2image(pred, filenames, output_folder)

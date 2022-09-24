import os
import sys

import imageio
import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torchvision.models import vgg16

from P2_dataloader import p2_dataset
from P2_models import U_Net


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

net = U_Net()
net.load_state_dict(torch.load('./P2_B_checkpoint/best_model.pth'))
net = net.to(device)

input_folder = sys.argv[1]
output_folder = sys.argv[2]

test_dataset = p2_dataset(input_folder, train=False)
test_loader = torch.utils.data.DataLoader(test_dataset, 16)

try:
    os.makedirs(output_folder, exist_ok=True)
except:
    pass
for x, filenames in test_loader:
    with torch.no_grad():
        x = x.to(device)
        out = net(x)
        print(out[0, :, 192, 192])
    pred = out.argmax(dim=1)

    print(filenames)
    pred2image(pred, filenames, output_folder)

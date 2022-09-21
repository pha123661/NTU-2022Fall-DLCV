import csv
import os
import sys

import numpy as np
import timm
import torch
from PIL import Image
from torchvision import transforms


class p1_test_dataset(torch.utils.data.Dataset):
    def __init__(self, folder_path):
        self.Images = []
        self.Transformation = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalization
            transforms.Normalize(mean, std),
        ])

        self.Filenames = [file for file in os.listdir(
            folder_path) if file.endswith('.png')]
        for filename in self.Filenames:
            self.Images.append(Image.open(os.path.join(
                folder_path, filename)).convert('RGB'))

    def __getitem__(self, index):
        return self.Transformation(self.Images[index]), self.Filenames[index]

    def __len__(self):
        return len(self.Images)


folder_path = sys.argv[1]
out_csv = sys.argv[2]

# read data
mean, std = [0.5077, 0.4813, 0.4312], [0.2000, 0.1986, 0.2034]
test_set = p1_test_dataset(folder_path)
test_loader = torch.utils.data.DataLoader(
    dataset=test_set, batch_size=32, shuffle=False)
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
# prepare model
net = timm.create_model('beit_base_patch16_224_in22k',
                        pretrained=False, num_classes=50)
net.load_state_dict(torch.load('P1_B_checkpoint/best_model.pth'))
net = net.to(device)
net.eval()

filenames = []
preds = []

with torch.no_grad():
    for x, names in test_loader:
        x = x.to(device)
        out = net(x)
        y_pred = torch.argmax(out, axis=1).flatten().detach().tolist()
        filenames.extend(names)
        preds.extend(y_pred)

filenames = np.array(filenames, dtype=str)
preds = np.array(preds, dtype=np.uint8)

s_idx = np.argsort(filenames)
filenames = filenames[s_idx]
preds = preds[s_idx]

os.makedirs(os.path.dirname(out_csv), exist_ok=True)

with open(out_csv, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(('filename', 'label'))
    for data in zip(filenames, preds):
        writer.writerow(data)

# gt = np.array([int(s.split('_')[0]) for s in filenames]).astype(np.uint8)
# print(np.sum(np.equal(gt, preds)) / len(gt))

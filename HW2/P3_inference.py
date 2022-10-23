import argparse
import csv
import pathlib

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from P3_SVHN_model import FeatureExtractor as SF
from P3_SVHN_model import LabelPredictor as SL
from P3_USPS_model import FeatureExtractor as UF
from P3_USPS_model import LabelPredictor as UL


class ImageFolderTestPNGDataset(Dataset):
    def __init__(self, path, transform):
        path = pathlib.Path(path)
        self.data = []
        for img in path.glob("*"):
            if img.is_file():
                self.data.append(img)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert('RGB')

        return self.transform(img), img_path.name


def main(args):
    USPS = 'usps' in str(args.input_dir)

    if USPS:
        mean, std = [0.2573, 0.2573, 0.2573], [0.3373, 0.3373, 0.3373]
    else:
        mean, std = [0.4413, 0.4458, 0.4715], [0.1169, 0.1206, 0.1042]
    print(f'mean, std = {mean}, {std}')

    dataset = ImageFolderTestPNGDataset(
        args.input_dir,
        transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ]),
    )
    test_loader = DataLoader(dataset, batch_size=64,
                             shuffle=False, num_workers=6)

    if USPS:
        print('loading usps model')
        F = UF().to(args.device)
        L = UL().to(args.device)
        F.load_state_dict(torch.load(
            args.usps_ckpt / "best_F.pth", map_location=args.device))
        L.load_state_dict(torch.load(
            args.usps_ckpt / "best_L.pth", map_location=args.device))
    else:
        print('loading svhn model')
        F = SF().to(args.device)
        L = SL().to(args.device)
        F.load_state_dict(torch.load(
            args.svhn_ckpt / "best_F.pth", map_location=args.device))
        L.load_state_dict(torch.load(
            args.svhn_ckpt / "best_L.pth", map_location=args.device))

    F.eval()
    L.eval()

    all_preds = []
    all_names = []
    for tgt_x, names in test_loader:
        tgt_x = tgt_x.to(args.device)

        with torch.no_grad():
            logits = L(F(tgt_x))
        pred = logits.argmax(-1).cpu().tolist()
        all_preds.extend(pred)
        all_names.extend(names)

    with open(args.out_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(('image_name', 'label'))
        for data in zip(all_names, all_preds):
            writer.writerow(data)

    # validation result
    try:
        if args.do_eval:
            cor = 0
            total = 0
            with open(f'hw2_data/digits/{"usps" if USPS else "svhn"}/val.csv', 'r') as file:
                reader = csv.reader(file)
                next(iter(reader))  # ignore first line
                for row in reader:
                    name, gt = row[0], int(row[1])
                    if gt == all_preds[all_names.index(name)]:
                        cor += 1
                    total += 1
            print(cor / total)
    except Exception as e:
        print(e)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-uw", "--usps_ckpt",
        help="directory of usps weight",
        type=pathlib.Path,
        default='./P3_USPS_ckpt/'
    )
    parser.add_argument(
        "-sw", "--svhn_ckpt",
        help="directory of svhn weight",
        type=pathlib.Path,
        default='./P3_SVHN_ckpt/'
    )
    parser.add_argument(
        "-i", "--input_dir",
        help="directory of input images",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "-o", "--out_csv",
        help="path to output csv",
        type=pathlib.Path,
        required=True,
    )
    parser.add_argument(
        "-d", "--device",
        help="device",
        type=torch.device,
        default='cuda' if torch.cuda.is_available() else 'cpu',
    )
    parser.add_argument(
        "-v", "--do_eval",
        action='store_true',
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    try:
        args.out_csv.parents[0].mkdir(exist_ok=True, parents=True)
    except:
        pass
    main(args)

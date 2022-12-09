import argparse
import csv
import json
import pathlib

import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms

from P2_dataloader import ImageFolderDataset
from P2_model import Classifier


@torch.no_grad()
def main(args):
    print('Loading dataset')
    image_size = 128
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    dataset = ImageFolderDataset(
        args.input_image_dir,
        transform=transform,
    )
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )
    with (args.ckpt_dir / "office_label2idx.json").open('r') as f:
        idx2label = {v: k for k, v in json.load(f).items()}

    print('Constructing model')
    model = Classifier(
        backbone=models.resnet50(weights=None),
        in_features=1000,
        n_class=len(idx2label),
        n_layers=args.n_layers,
        dropout=args.dropout,
        hidden_size=args.hidden_size,
    ).to(args.device)
    model.load_state_dict(torch.load(
        args.ckpt_dir / "classifier.pth", map_location=args.device))
    model.eval()

    print('Predicting')
    filename2pred_class = dict()
    for data in dataloader:
        img = data['img'].to(args.device)
        logits = model(img)
        y_pred = torch.argmax(logits, dim=1)
        for filename, pred in zip(data['filename'], y_pred):
            class_name = idx2label[pred.item()]
            filename2pred_class[filename] = class_name

    print('Flushing output')
    # write output
    with args.input_csv.open('r') as in_f:
        with args.output_csv.open('w') as out_f:
            reader = csv.reader(in_f)
            next(iter(reader))  # skip header
            writer = csv.writer(out_f)
            writer.writerow(('id', 'filename', 'label'))

            for id, filename, _ in reader:
                writer.writerow((
                    id,
                    filename,
                    filename2pred_class[filename]
                ))

    print('Done')


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument('--input_image_dir',
                        type=pathlib.Path, required=True)
    parser.add_argument("--input_csv", type=pathlib.Path, required=True)
    parser.add_argument('--output_csv', type=pathlib.Path, required=True)
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")

    # Model args
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--hidden_size', type=int, default=256)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    try:
        args.output_csv.parent.mkdir(exist_ok=True, parents=True)
    except Exception as e:
        print('Error:', e)
    main(args)

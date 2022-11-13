import argparse
import json
import pathlib

import timm


def main(args):
    config = json.load((args.ckpt_dir / "model_config.json").open(mode='r'))
    timm.create_model(config['encoder'], pretrained=True, num_classes=0)


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_dir",
                        type=pathlib.Path, default="./P2_ckpt")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args)

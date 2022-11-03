import argparse
import json
import os
import pathlib
from collections import defaultdict

import clip
import language_evaluation
import torch
from PIL import Image
from tqdm.auto import tqdm


def main(args):
    preds = json.load(args.pred_json.open(mode='r'))
    # # CLIP score
    # model, image_process = clip.load(args.CLIP, device=args.device)
    # clip_scores = []
    # for image_name, text in tqdm(preds.items()):
    #     image = Image.open(args.image_dir / f"{image_name}.jpg").convert('RGB')
    #     image = image_process(image).unsqueeze(0).to(args.device)
    #     text = clip.tokenize(text).to(args.device)

    #     with torch.no_grad():
    #         image_features = model.encode_image(image)
    #         image_features /= image_features.norm(dim=-1, keepdim=True)
    #         text_features = model.encode_text(text)
    #         text_features /= text_features.norm(dim=-1, keepdim=True)

    #     sim = image_features @ text_features.T
    #     score = 2.5 * max(sim.item(), 0)
    #     clip_scores.append(score)

    # clip_score = sum(clip_scores) / len(clip_scores)
    # print(f'clip score={clip_score}')

    # CIDEr score
    evaluator = language_evaluation.CocoEvaluator(coco_types=['CIDEr'])
    info = json.load(args.info_json.open(mode='r'))
    annotations = defaultdict(list)
    for data in info['annotations']:
        annotations[data['image_id']].append(data['caption'])
    img2id = {os.path.splitext(data['file_name'])[0]: data['id']
              for data in info['images']}

    CIDEr_scores = []
    for image_name, text in preds.items():
        ans = annotations[img2id[image_name]]
        score = evaluator.run_evaluation(
            text, ans)['CIDEr']
        CIDEr_scores.append(score)
        print(score)
    CIDEr_score = sum(CIDEr_scores) / len(CIDEr_scores)
    print(f'CIDEr score={CIDEr_score}')


def parse():
    parser = argparse.ArgumentParser()

    # Environment
    parser.add_argument("--pred_json", type=pathlib.Path, required=True)
    parser.add_argument("--image_dir", type=pathlib.Path,
                        default="./hw3_data/p2_data/images/val")
    parser.add_argument('--CLIP', default='RN101')
    parser.add_argument('--device', type=torch.device,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--info_json', type=pathlib.Path,
                        default='hw3_data/p2_data/val.json')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    main(args)

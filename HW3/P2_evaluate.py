import json
import os
from argparse import ArgumentParser
from collections import defaultdict

import clip
import language_evaluation
import torch
from numpy import dot
from numpy.linalg import norm
from PIL import Image


def readJSON(file_path):
    try:
        with open(file_path) as f:
            data = json.load(f)
        return data
    except:
        return None


def getGTCaptions(annotations):
    img_id_to_name = {}
    for img_info in annotations["images"]:
        img_name = img_info["file_name"].replace(".jpg", "")
        img_id_to_name[img_info["id"]] = img_name

    img_name_to_gts = defaultdict(list)
    for ann_info in annotations["annotations"]:
        img_id = ann_info["image_id"]
        img_name = img_id_to_name[img_id]
        img_name_to_gts[img_name].append(ann_info["caption"])
    return img_name_to_gts


class CIDERScore:
    def __init__(self):
        self.evaluator = language_evaluation.CocoEvaluator(coco_types=[
                                                           "CIDEr"])

    def __call__(self, predictions, gts):
        """
        Input:
            predictions: dict of str
            gts:         dict of list of str
        Return:
            cider_score: float
        """
        # Collect predicts and answers
        predicts = []
        answers = []
        for img_name in predictions.keys():
            predicts.append(predictions[img_name])
            answers.append(gts[img_name])

        # Compute CIDEr score
        results = self.evaluator.run_evaluation(predicts, answers)
        return results['CIDEr']


class CLIPScore:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.model.eval()

    def __call__(self, predictions, images_root):
        """
        Input:
            predictions: dict of str
            images_root: str
        Return:
            clip_score: float
        """
        total_score = 0.

        for img_name, pred_caption in predictions.items():
            image_path = os.path.join(images_root, f"{img_name}.jpg")
            image = Image.open(image_path).convert("RGB")

            total_score += self.getCLIPScore(image, pred_caption)
        return total_score / len(predictions)

    def getCLIPScore(self, image, caption):
        """
        This function computes CLIPScore based on the pseudocode in the slides.
        Input:
            image: PIL.Image
            caption: str
        Return:
            cilp_score: float
        """
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        caption = clip.tokenize(caption).to(self.device)
        image_embed = self.model.encode_image(image)[0].detach().cpu()
        caption_embed = self.model.encode_text(caption)[0].detach().cpu()
        return 2.5 * max(dot(image_embed, caption_embed) / (norm(image_embed) * norm(caption_embed)), 0)


def main(args):
    # Read data
    predictions = readJSON(args.pred_file)
    annotations = readJSON(args.annotation_file)

    # Preprocess annotation file
    gts = getGTCaptions(annotations)

    # Check predictions content is correct
    assert type(predictions) is dict
    assert set(predictions.keys()) == set(gts.keys())
    assert all([type(pred) is str for pred in predictions.values()])

    # CIDErScore
    cider_score = CIDERScore()(predictions, gts)

    # CLIPScore
    clip_score = CLIPScore()(predictions, args.images_root)

    print(f"CIDEr: {cider_score} | CLIPScore: {clip_score}")


if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("-p", "--pred_file",
                        default="./pred.json", help="Prediction json file")
    parser.add_argument(
        "--images_root", default="hw3_data/p2_data/images/val/", help="Image root")
    parser.add_argument("--annotation_file",
                        default="hw3_data/p2_data/val.json", help="Annotation json file")

    args = parser.parse_args()

    main(args)

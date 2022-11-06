#!/bin/bash

# TODO - run your inference Python3 code
python3 P2_inference.py --image_dir=$1 --output_json=$2 --tokenizer="./P2_ckpt/caption_tokenizer.json"
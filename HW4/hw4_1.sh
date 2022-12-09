#!/bin/bash
python3 P1_inference.py --config=./hotdog.py --render_only --render_test --dump_images --input_json=$1 --output_dir=$2
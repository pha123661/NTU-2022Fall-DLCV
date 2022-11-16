#!/bin/bash

gdown --folder https://drive.google.com/drive/folders/18zVD_afRmTll44S91EOkB20ow03flDF7?usp=sharing
python -c "import clip; clip.load('ViT-L/14')"

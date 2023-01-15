export PYTHONPATH=./
python tool/val_miou.py \
    --sub_root "/tmp3/ycwei/test_inference/2022_fall_DLCV_final/point-transformer/submission" \
    --gt_root "/tmp3/ycwei/2022_fall_DLCV_final/point-transformer/dataset/scannet200" \
    --split_dir "/tmp3/ycwei/2022_fall_DLCV_final/point-transformer/dataset/scannet200/val_split.txt"
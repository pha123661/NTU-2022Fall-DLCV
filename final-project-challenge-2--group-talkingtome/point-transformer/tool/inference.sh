#!/bin/sh

export PYTHONPATH=./
# eval "$(conda shell.bash hook)"
PYTHON=python

dataset=$1
exp_name=$2

exp_version=$3
data_dir=$4
split=$5
output_dir=$6

exp_dir=${exp_version}/${dataset}/${exp_name}
model_dir=${exp_dir}/model
result_dir=${exp_dir}/result
# config=config/${dataset}/${dataset}_${exp_name}.yaml
config=${exp_dir}/scannet200_pointtransformer_repro.yaml

# mkdir -p ${result_dir}/last
# mkdir -p ${result_dir}/best
# cp ${config} tool/test.sh tool/${TEST_CODE} ${exp_dir}

#: '
$PYTHON -u tool/inference.py \
  --config=${config} \
  model_path ${model_dir}/model_best.pth \
  data_dir ${data_dir} \
  split_txt_dir ${split} \
  output_dir ${output_dir} \
  2>&1 | tee ${exp_dir}/test_best-$now.log
  # save_folder ${result_dir}/best \
#'
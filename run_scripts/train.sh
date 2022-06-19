#!/bin/bash

wandb=''

while getopts e:c:w: flag
do
    case "${flag}" in
        e) env_name=${OPTARG};;
        c) case=${OPTARG};;
        w) wandb='--wandb';;
    esac
done

python3 main.py --env $env_name --case $case --opr train --force \
  --num_gpus 2 --num_cpus 96 --cpu_actor 14 --gpu_actor 10 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  $wandb \
  --object_store_memory 32000000000 # 32GB

#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
sudo mount /dev/shm

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
  --num_gpus 0 --num_cpus 14 --cpu_actor 4 --gpu_actor 0 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  $wandb \
  --object_store_memory 10000000000 # 20GB

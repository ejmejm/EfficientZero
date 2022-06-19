# sudo mount /dev/shm

python3 main.py --env BreakoutNoFrameskip-v4 --case atari_fast --opr train --force \
  --num_gpus 2 --num_cpus 96 --cpu_actor 14 --gpu_actor 10 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  --object_store_memory 20000000000 # 20GB

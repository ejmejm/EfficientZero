sudo mount /dev/shm

python main.py --env 'MiniGrid-MultiRoom-N2-S4-v0' --case minigrid_debug --opr train --force \
  --num_gpus 1 --num_cpus 14 --cpu_actor 6 --gpu_actor 4 \
  --seed 0 \
  --use_priority \
  --use_max_priority \
  --amp_type 'torch_amp' \
  --info 'EfficientZero-V1' \
  --object_store_memory 20000000000 # 20GB
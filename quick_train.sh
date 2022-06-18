sudo mount /dev/shm

python main.py --env BreakoutNoFrameskip-v4 --case atari --opr train --amp_type torch_amp --num_gpus 1 --num_cpus 10 --cpu_actor 1 --gpu_actor 1 --object_store_memory 20000000000 --force
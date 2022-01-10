export FLAGS_fraction_of_gpu_memory_to_use=0.8
export CUDA_VISIBLE_DEVICES=1
nohup python -u train.py --config=tf/full_strs/full_strs --use_gpu=True --data_root=./data & 

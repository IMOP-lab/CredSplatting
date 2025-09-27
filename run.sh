# torchrun --nproc_per_node=1 --master_port=29501 run.py --type evaluate --cfg_file configs/credsplatting/dtu_pretrain.yaml credsplatting.cas_config.render_if True,True credsplatting.cas_config.volume_planes 64,8 credsplatting.eval_depth True gpus 0, 

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/llff_eval.yaml distributed True gpus 0,

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/tnt_eval.yaml distributed True gpus 0,

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/nerf_eval.yaml distributed True gpus 0,

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/dtu_pretrain.yaml distributed True gpus 0, test_dataset.scene scan28

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/dtu_pretrain.yaml distributed True gpus 0, credsplatting.cas_config.render_if False,True

# torchrun --nproc_per_node=1  --master_port=29500 train_net.py --cfg_file configs/credsplatting/colmap_train.yaml train.batch_size 2 credsplatting.eval_depth True gpus 0,

# torchrun --nproc_per_node=1 --master_port=29500 train_net.py --cfg_file configs/credsplatting/colmap_train.yaml distributed True gpus 0, train.batch_size 1

# torchrun --nproc_per_node=2 --master_port=29500 train_net.py --cfg_file configs/credsplatting/tnt/Train.yaml distributed True gpus 0,1 train.batch_size 1
# torchrun --nproc_per_node=2 --master_port=29500 train_net.py --cfg_file configs/credsplatting/tnt/Truck.yaml distributed True gpus 0,1 train.batch_size 1

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/tnt/Truck.yaml distributed True gpus 0,

# python lib/colmap/imgs2poses.py -s datasets/syn/Frames_S1

torchrun --nproc_per_node=2 --master_port=29500 train_net.py --cfg_file configs/credsplatting/multi_dataset_pretrain.yaml distributed True gpus 0,1 train.batch_size 2

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/multi_dataset_pretrain.yaml distributed True gpus 0,


# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/llff_eval.yaml distributed True gpus 0,

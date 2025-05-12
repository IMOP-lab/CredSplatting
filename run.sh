# torchrun --nproc_per_node=1 --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/dtu_pretrain.yaml credsplatting.cas_config.render_if True,True credsplatting.cas_config.volume_planes 64,8 credsplatting.eval_depth True gpus 0,

torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/llff_eval.yaml distributed True gpus 0,

# torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/tnt_eval.yaml distributed True gpus 0,
<div align="center">
  <h1>CredSplatting: Perception Credence Oriented Feed-forward 3D Gaussian Splatting from Scalable Views</h1>
<!--   <h2>NeurIPS 2024 (poster)</h2> -->
</div>

## 🚀Datasets
Download DTU data and Depth raw. Unzip and organize them as:
```
yours_datasets_path
    ├── dtu                   
        ├── Cameras                
        ├── Depths   
        ├── Depths_raw
        └── Rectified
```
Download Real Forward-facing, and Tanks and Temples datasets.
Then modify the dataset path in the YAML files under the configs folder.

## 🚀Evaluation
### Evaluation on DTU
```
torchrun --nproc_per_node=1 --master_port=29501 run.py --type evaluate --cfg_file configs/credsplatting/dtu_pretrain.yaml credsplatting.cas_config.render_if True,True credsplatting.cas_config.volume_planes 64,8 credsplatting.eval_depth True gpus 0, 
```

### Evaluation on Real Forward-facing
```
torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/llff_eval.yaml distributed True gpus 0, credsplatting.cas_config.render_if False,True
```

### Evaluation on Tanks and Temples datasets
```
torchrun --nproc_per_node=1  --master_port=29500 run.py --type evaluate --cfg_file configs/credsplatting/tnt_eval.yaml distributed True gpus 0, credsplatting.cas_config.render_if False,True
```

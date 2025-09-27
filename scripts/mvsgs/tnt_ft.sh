export CUDA_VISIBLE_DEVICES=0
data_dir="/home/lab2/2T/CredSplatting/datasets/tnt"
dir_ply="credsplatting_pointcloud"
scenes=(Train Truck)
# scenes=(Truck)
iter=5000

python run.py --type evaluate --cfg_file configs/credsplatting/tnt_eval.yaml save_ply True dir_ply $dir_ply

for scene in ${scenes[@]}
do  
python lib/train.py  --eval --iterations $iter -s $data_dir/$scene -p $dir_ply
python lib/render.py -c -m output/$scene --iteration $iter -p $dir_ply
python lib/metrics.py -m output/$scene
done

python lib/utils/read_json.py --root output --scenes ${scenes[@]} --iter $iter
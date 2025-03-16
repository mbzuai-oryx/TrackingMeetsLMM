LLAMA_PATH="/data1/DriveLM/llama_wts"
PRETRAINED_PATH="/home/fahad/ayesha/DriveLM/trained_weights/pretrained_LORA-BIAS-7B.pth" 
CONFIG="/home/fahad/ayesha/DriveLM/challenge/llama_adapter_v2_multimodal7b/finetune_data_config.yaml"
OUTPUT_DIR="/data2/DriveLM/CARLA/baseline"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="2,3" python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 2 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 --start_epoch 0 \
 &>> "$OUTPUT_DIR"/output.log&

#  CUDA_VISIBLE_DEVICES="0,1" python demo.py --llama_dir /data1/DriveLM/llama_wts/ --checkpoint /data2/DriveLM/CARLA/baseline/converted-checkpoint.pth --data /data2/DriveLM/CARLA/data/test_llama.json --output /data2/DriveLM/CARLA/baseline/test_output.json --batch_size 4 --num_processes 2
#  CUDA_VISIBLE_DEVICES="2,3" python demo.py --llama_dir /data1/DriveLM/llama_wts/ --checkpoint /data2/DriveLM/CARLA/traj_5_all/converted-checkpoint.pth --data /home/fahad/ayesha/DriveLM/carla/data/val_llama.json --output /data2/DriveLM/CARLA/traj_5_all/val_output.json --batch_size 4 --num_processes 2


 
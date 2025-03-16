#!/usr/bin/bash

LLAMA_PATH="$1"
CONFIG="$2"
OUTPUT_DIR="$3"

LLAMA_PATH="/data1/DriveLM/llama_wts"
PRETRAINED_PATH="/home/fahad/ayesha/DriveLM/trained_weights/pretrained_LORA-BIAS-7B.pth" # path to pre-trained checkpoint
CONFIG="/home/fahad/ayesha/DriveLM/challenge/llama_adapter_v2_multimodal7b/finetune_data_config.yaml"
OUTPUT_DIR="/data2/DriveLM/multimodal_traj_pretrain_ego"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="2,3" python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 --use_env \
 main_pretrain.py --data_config "$CONFIG" --batch_size 2 \
--epochs 150 --split_epoch 50 --warmup_epochs 5 --blr 1.0e-4 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 &>> "$OUTPUT_DIR"/output.log &
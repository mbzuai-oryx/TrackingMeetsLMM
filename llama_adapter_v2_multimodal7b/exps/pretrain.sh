#!/usr/bin/bash

LLAMA_PATH="$1"
CONFIG="$2"
OUTPUT_DIR="$3"

mkdir -p "$OUTPUT_DIR"

CUDA_VISIBLE_DEVICES="2,3" python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 --use_env \
 main_pretrain.py --data_config "$CONFIG" --batch_size 2 \
--epochs 150 --split_epoch 50 --warmup_epochs 5 --blr 1.0e-4 --weight_decay 0.05 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 &>> "$OUTPUT_DIR"/output.log &

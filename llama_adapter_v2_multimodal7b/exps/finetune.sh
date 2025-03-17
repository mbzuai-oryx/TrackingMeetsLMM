
mkdir -p "$OUTPUT_DIR"

python -u -m torch.distributed.launch --master_port=1112 --nproc_per_node=2 --use_env \
 main_finetune.py --data_config "$CONFIG" --batch_size 2 \
 --epochs 4 --warmup_epochs 1 --blr 10e-4 --weight_decay 0.02 \
 --llama_path "$LLAMA_PATH" \
 --output_dir "$OUTPUT_DIR" \
 --pretrained_path "$PRETRAINED_PATH" \
 --start_epoch 0 \
 &>> "$OUTPUT_DIR"/output.log&

 

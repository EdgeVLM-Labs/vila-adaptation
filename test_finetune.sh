#!/bin/bash

#################################################################
# Test Fine-tuning Script for VILA with Custom Video Dataset
#################################################################
# This script performs a quick test fine-tune on a small subset
# to verify your custom dataset works before full training
#################################################################

set -e  # Exit on error

echo "=========================================="
echo "VILA Fine-tuning Test Script"
echo "=========================================="

# Configuration
MODEL_PATH=${1:-"Efficient-Large-Model/VILA1.5-3b"}  # Start with smaller model for testing
DATA_MIXTURE=${2:-"my_video_sft"}  # Your custom dataset name from registry
OUTPUT_DIR=${3:-"runs/test/my_video_finetune_test"}
NUM_EPOCHS=${4:-1}
MAX_STEPS=${5:-10}  # Only run 10 steps for quick test

echo "Model Path: $MODEL_PATH"
echo "Data Mixture: $DATA_MIXTURE"
echo "Output Directory: $OUTPUT_DIR"
echo "Test Steps: $MAX_STEPS"
echo ""

# Check if model exists locally or needs download
if [[ $MODEL_PATH == *"/"* ]] && [[ ! -d "$HOME/.cache/huggingface/hub"*"$MODEL_PATH"* ]]; then
    echo "Model not found locally. It will be downloaded automatically..."
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "=========================================="
echo "Starting test fine-tune..."
echo "=========================================="

# Set environment variable to load custom dataset registry
export VILA_DATASETS="default,my_video"

# Run a minimal training test with reduced batch size and steps
python llava/train/train_mem.py \
    --model_name_or_path "$MODEL_PATH" \
    --version vicuna_v1 \
    --data_mixture "$DATA_MIXTURE" \
    --vision_tower google/siglip-so400m-patch14-384 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy no \
    --save_strategy steps \
    --save_steps 5 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0.0 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --report_to none \
    --max_steps $MAX_STEPS \
    --fp16 False

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo "If the script completed without errors, your dataset is properly configured!"
echo "Trained checkpoint saved to: $OUTPUT_DIR"
echo ""
echo "Next steps:"
echo "1. Check the logs above for any data loading issues"
echo "2. Verify the loss is decreasing (check wandb or logs)"
echo "3. If everything looks good, proceed with full training"
echo ""
echo "To run full training, use the official script:"
echo "  bash scripts/NVILA-Lite/sft.sh <pretrained_model_path> $DATA_MIXTURE"
echo "=========================================="

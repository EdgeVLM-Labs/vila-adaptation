#!/bin/bash
set -e

# ================================
# Defaults (can be overridden)
# ================================
DEFAULT_RUN_NAME=${DEFAULT_RUN_NAME:-"NVILA-2B-finetune"}
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${DEFAULT_GLOBAL_TRAIN_BATCH_SIZE:-64}
DEFAULT_GRADIENT_ACCUMULATION_STEPS=${DEFAULT_GRADIENT_ACCUMULATION_STEPS:-8}
PER_DEVICE_EVAL_BATCH_SIZE=${PER_DEVICE_EVAL_BATCH_SIZE:-8}

# Eval params
EVAL_DATA_MIXTURE=${EVAL_DATA_MIXTURE:-""}
EVAL_STEPS=${EVAL_STEPS:-50}
EVALUATION_STRATEGY=${EVALUATION_STRATEGY:-"no"}

STAGE_PATH=${1}
DATA_MIXTURE=${2}
OUTPUT_DIR=${3}

# ================================
# Logging
# ================================
mkdir -p logs
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="logs/finetune_${TIMESTAMP}.log"

exec > >(tee -a "$LOG_FILE") 2>&1

echo "========================================="
echo "VILA Finetuning"
echo "Started at: $(date)"
echo "========================================="

# ================================
# Environment
# ================================
source scripts/setups/train.sh

# Override GPU count if provided
if [ -n "$DEFAULT_GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
fi

# ================================
# Batch size math (CORRECT)
# ================================
GLOBAL_TRAIN_BATCH_SIZE=$DEFAULT_GLOBAL_TRAIN_BATCH_SIZE
GRADIENT_ACCUMULATION_STEPS=$DEFAULT_GRADIENT_ACCUMULATION_STEPS

PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / NNODES / GPUS_PER_NODE / GRADIENT_ACCUMULATION_STEPS))

echo "NNODES = $NNODES"
echo "GPUS_PER_NODE = $GPUS_PER_NODE"
echo "GLOBAL_TRAIN_BATCH_SIZE = $GLOBAL_TRAIN_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS = $GRADIENT_ACCUMULATION_STEPS"
echo "PER_DEVICE_TRAIN_BATCH_SIZE = $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "PER_DEVICE_EVAL_BATCH_SIZE = $PER_DEVICE_EVAL_BATCH_SIZE"

# ================================
# Training
# ================================
torchrun \
  --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
  --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
  llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path $STAGE_PATH \
    --data_mixture $DATA_MIXTURE \
    --eval_data_mixture $EVAL_DATA_MIXTURE \
    --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
    --mm_vision_select_feature cls_patch \
    --mm_projector mlp_downsample_3x3_fix \
    --mm_vision_select_layer -2 \
    --tune_vision_tower False \
    --tune_mm_projector True \
    --tune_language_model True \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio dynamic \
    --bf16 True \
    --tf32 True \
    --output_dir $OUTPUT_DIR/model \
    --num_train_epochs 3 \
    --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
    --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy $EVALUATION_STRATEGY \
    --eval_steps $EVAL_STEPS \
    --save_strategy steps \
    --save_steps 20 \
    --save_total_limit 3 \
    --learning_rate 5e-5 \
    --warmup_ratio 0.1 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 5 \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 2 \
    --num_video_frames 4 \
    --fps 1.0 \
    --downsample_video True \
    --vflan_no_system_prompt True \
    --group_by_modality_length True \
    --lora_enable True \
    --lora_llm True \
    --lora_r 64 \
    --lora_alpha 128 \
    --lora_dropout 0.05 \
    --lora_bias none \
    --report_to wandb

echo "========================================="
echo "Training completed at: $(date)"
echo "Model saved to: $OUTPUT_DIR/model"
echo "========================================="

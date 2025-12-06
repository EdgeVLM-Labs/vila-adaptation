#!/bin/bash

# Use environment variables if set, otherwise use defaults
DEFAULT_RUN_NAME=${DEFAULT_RUN_NAME:-"vila-qwen2-vl-7b-sft"}
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=${DEFAULT_GLOBAL_TRAIN_BATCH_SIZE:-2048}
DEFAULT_GRADIENT_ACCUMULATION_STEPS=${DEFAULT_GRADIENT_ACCUMULATION_STEPS:-2}

STAGE_PATH=${1:-"runs/train/nvila-8b-pretrain/model"}
DATA_MIXTURE=${2:-"nvila-pretrain"}
OUTPUT_DIR=${3:-"runs/train/nvila-8b-sft"}

source scripts/setups/train.sh

# Override GPU count if DEFAULT_GPUS_PER_NODE is set
if [ ! -z "$DEFAULT_GPUS_PER_NODE" ]; then
    GPUS_PER_NODE=$DEFAULT_GPUS_PER_NODE
    echo "Overriding GPUS_PER_NODE = $GPUS_PER_NODE"
    # Recalculate batch size with new GPU count
    PER_DEVICE_TRAIN_BATCH_SIZE=$((GLOBAL_TRAIN_BATCH_SIZE / NNODES / GPUS_PER_NODE / GRADIENT_ACCUMULATION_STEPS))
    echo "Recalculated PER_DEVICE_TRAIN_BATCH_SIZE = $PER_DEVICE_TRAIN_BATCH_SIZE"
fi

STAGE2_PATH=$1

torchrun \
    --nnodes=$NNODES --nproc_per_node=$GPUS_PER_NODE --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \
    llava/train/train_mem.py \
        --deepspeed scripts/zero3.json \
        --model_name_or_path $STAGE_PATH \
        --data_mixture $DATA_MIXTURE \
        --vision_tower Efficient-Large-Model/paligemma-siglip-so400m-patch14-448 \
        --mm_vision_select_feature cls_patch \
        --mm_projector mlp_downsample_3x3_fix \
        --tune_vision_tower False \
        --tune_mm_projector True \
        --tune_language_model True \
        --mm_vision_select_layer -2 \
        --mm_use_im_start_end False \
        --mm_use_im_patch_token False \
        --image_aspect_ratio dynamic \
        --bf16 True \
        --output_dir $OUTPUT_DIR/model \
        --num_train_epochs 1 \
        --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
        --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
        --evaluation_strategy no \
        --save_strategy steps \
        --save_steps 100 \
        --save_total_limit 1 \
        --learning_rate 3e-5 \
        --weight_decay 0. \
        --warmup_ratio 0.03 \
        --lr_scheduler_type cosine \
        --logging_steps 1 \
        --model_max_length 2048 \
        --gradient_checkpointing True \
        --dataloader_num_workers 4 \
        --num_video_frames 4 \
        --fps 1.0 \
        --downsample_video True \
        --vflan_no_system_prompt True \
        --group_by_modality_length True \
        --lora_enable True \
        --lora_llm True \
        --lora_r 16 \
        --lora_alpha 32 \
        --lora_dropout 0.05 \
        --lora_bias none \
        --report_to wandb

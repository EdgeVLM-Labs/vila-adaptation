# Optimized for 48GB GPU - can handle larger batch sizes
export DEFAULT_RUN_NAME="VILA-3B-finetune"
export DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=64
export DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
export DEFAULT_GPUS_PER_NODE=1

export WANDB_PROJECT="VILA"
export WANDB_NAME="finetune-vila1.5-3b-qved"

bash scripts/NVILA-Lite/sft.sh \
    Efficient-Large-Model/VILA1.5-3b \
    QVED-dataset \
    runs/train/vila-3b-exercise-finetune

# Upload to Hugging Face after training
echo "Training completed. Uploading model to Hugging Face..."
huggingface-cli upload EdgeVLM-Labs/VILA-3B-QVED-Finetune runs/train/vila-3b-exercise-finetune/model --include="*"
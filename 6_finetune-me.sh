# Optimized for 48GB GPU - can handle larger batch sizes
export DEFAULT_RUN_NAME="VILA-3B-finetune-2"
export DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=32
export DEFAULT_GRADIENT_ACCUMULATION_STEPS=4
export DEFAULT_GPUS_PER_NODE=1

export WANDB_PROJECT="VILA"
export WANDB_NAME="finetune-vila1.5-3b-qved"

# Add validation dataset support
export EVAL_DATA_MIXTURE="QVED-dataset-val"  # Validation dataset
export EVAL_STEPS=30  # Evaluate every 30 steps
export SAVE_STEPS=30  # Save every 30 steps
export EVALUATION_STRATEGY="steps"  # Enable evaluation
export PER_DEVICE_EVAL_BATCH_SIZE=8  # Evaluation batch size

bash scripts/NVILA-Lite/sft.sh \
    Efficient-Large-Model/VILA1.5-3b \
    QVED-dataset \
    runs/train/vila-3b-exercise-finetune

# Upload to Hugging Face after training
echo "Training completed. Uploading model to Hugging Face..."
huggingface-cli upload EdgeVLM-Labs/VILA1.5-3B-finetuned-500 runs/train/vila-3b-exercise-finetune/model --include="*"
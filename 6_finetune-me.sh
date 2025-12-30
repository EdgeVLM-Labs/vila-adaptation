# Optimized for 48GB GPU
export DEFAULT_RUN_NAME="NVILA-2B-finetune"
export DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=64
export DEFAULT_GRADIENT_ACCUMULATION_STEPS=8
export DEFAULT_GPUS_PER_NODE=1

export WANDB_ENTITY="fyp-21"
export WANDB_PROJECT="NVILA"
export WANDB_NAME="finetune-nvila-2b-qved"

# Validation
export EVAL_DATA_MIXTURE="QVED-dataset-val"
export EVAL_STEPS=50
export EVALUATION_STRATEGY="steps"
export PER_DEVICE_EVAL_BATCH_SIZE=8   # 👈 add this

bash scripts/NVILA-Lite/sft.sh \
    Efficient-Large-Model/NVILA-Lite-2B \
    QVED-dataset \
    runs/train/nvila-2b-exercise-finetune
    
# Upload to Hugging Face after training
echo "Training completed. Uploading model to Hugging Face..."
huggingface-cli upload EdgeVLM-Labs/NVILA-Lite-2B-finetuned-2000 runs/train/nvila-2b-exercise-finetune/model --include="*"
echo "Model upload completed."
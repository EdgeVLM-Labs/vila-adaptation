# reduce the global bs or increase the accum steps if facing OOM issues
DEFAULT_RUN_NAME="NVILA-Lite-8B-finetune-trial" \
DEFAULT_GLOBAL_TRAIN_BATCH_SIZE=4 \
DEFAULT_GRADIENT_ACCUMULATION_STEPS=2 \
DEFAULT_GPUS_PER_NODE=1 \
    bash scripts/NVILA-Lite/sft.sh \
        Efficient-Large-Model/VILA1.5-3b \
        SampleQA
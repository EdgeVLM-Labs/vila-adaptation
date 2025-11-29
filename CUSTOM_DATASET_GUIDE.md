# Custom Dataset Fine-tuning Guide for VILA

This guide explains how to fine-tune VILA with your own video/image dataset.

## Quick Start

You already have a custom dataset at `data/my_video_sft/train.jsonl`. Here's how to test and fine-tune with it:

### 1. Test Your Dataset (Recommended First Step)

Run a quick test to verify your dataset loads correctly:

```bash
chmod +x test_finetune.sh
./test_finetune.sh
```

This will:
- Run only 10 training steps
- Use a small model (VILA1.5-3b)
- Verify your data format is correct
- Complete in a few minutes

### 2. Full Fine-tuning

Once the test passes, run full fine-tuning:

```bash
# Option A: Use the official script with your dataset
bash scripts/NVILA-Lite/sft.sh <path_to_pretrained_model> my_video_sft

# Option B: Start from a base model
bash scripts/NVILA-Lite/sft.sh Efficient-Large-Model/VILA1.5-3b my_video_sft
```

## Dataset Format

Your dataset follows the VILA video format in `train.jsonl`:

```jsonl
{
  "id": "unique_video_id",
  "video": "/absolute/path/to/video.mp4",
  "num_frames": 10,
  "fps": 30,
  "conversations": [
    {
      "from": "user",
      "value": "Your question or instruction here"
    },
    {
      "from": "assistant", 
      "value": "Expected response here"
    }
  ]
}
```

### Important Fields:

- **id**: Unique identifier for each sample
- **video**: Absolute path to video file (or relative to `media_dir` in registry)
- **num_frames**: Number of frames to extract (optional, defaults to model config)
- **fps**: Frame rate for sampling (optional)
- **conversations**: List of turn-by-turn conversation

## Dataset Registration

Your dataset is already registered in `llava/data/registry/datasets/my_video.yaml`:

```yaml
my_video_sft:
  type: jsonl_conversation_video
  ann_path: /data/my_video_sft/train.jsonl
  media_dir: /
```

### Customization Options:

```yaml
my_video_sft:
  type: jsonl_conversation_video
  ann_path: /path/to/train.jsonl
  media_dir: /base/path/for/videos
  video_num_frames: 32          # Override frames per video
  frame_sampling: uniform        # 'uniform', 'rand', or 'stride'
  max_samples: 1000             # Limit samples for debugging
```

## Creating Your Own Dataset

### Step 1: Prepare Your Data

Create a JSONL file with one sample per line:

```python
import json

data = [
    {
        "id": "sample_001",
        "video": "/data/videos/exercise1.mp4",
        "num_frames": 10,
        "conversations": [
            {"from": "user", "value": "Describe this exercise."},
            {"from": "assistant", "value": "This is a high knees exercise..."}
        ]
    },
    # Add more samples...
]

with open("my_dataset.jsonl", "w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")
```

### Step 2: Register Your Dataset

Create `llava/data/registry/datasets/my_custom.yaml`:

```yaml
my_custom_dataset:
  type: jsonl_conversation_video  # or jsonl_conversation_image for images
  ann_path: /path/to/my_dataset.jsonl
  media_dir: /path/to/media/folder
```

### Step 3: (Optional) Create a Mixture

In `llava/data/registry/mixtures.yaml`, combine multiple datasets:

```yaml
my_training_mix:
  - my_video_sft
  - my_custom_dataset
  - sharegpt4v_gpt4_100k@0.5  # Use 50% of this dataset
```

### Step 4: Test and Train

```bash
# Test with your dataset
./test_finetune.sh Efficient-Large-Model/VILA1.5-3b my_custom_dataset

# Full training
bash scripts/NVILA-Lite/sft.sh <base_model> my_custom_dataset
```

## Dataset Types

VILA supports multiple dataset types:

| Type | Use Case | Example |
|------|----------|---------|
| `jsonl_conversation_video` | Video Q&A | Your exercise videos |
| `jsonl_conversation_image` | Image Q&A | Single image tasks |
| `webdataset` | Large-scale pretraining | TAR archives |
| `hf_parquet` | Hugging Face datasets | Cloud datasets |

## Hardware Requirements

### For Testing (test_finetune.sh)
- **GPU**: 1x GPU with 16GB+ VRAM (RTX 3090, A10, etc.)
- **RAM**: 32GB+
- **Storage**: 50GB for model + your dataset

### For Full Fine-tuning (official scripts)
- **Recommended**: 8x A100 (80GB) - as mentioned in README
- **Minimum**: 4x A100 (40GB) with DeepSpeed ZeRO-3
- **Budget Option**: 
  - Use smaller model (VILA1.5-3b instead of 8b)
  - Reduce batch size and increase gradient accumulation
  - Enable more aggressive memory optimizations

## Training Parameters Explained

```bash
--per_device_train_batch_size 1    # Batch per GPU (reduce if OOM)
--gradient_accumulation_steps 8     # Accumulate before update (increase if reducing batch size)
--num_train_epochs 1                # Full passes through data
--learning_rate 2e-5                # Fine-tuning LR (lower than pretraining)
--model_max_length 2048             # Max tokens (4096 for long videos)
--tune_vision_tower False           # Usually keep frozen for fine-tuning
--tune_mm_projector True            # Update vision-language bridge
--tune_language_model True          # Update LLM
```

## Monitoring Training

### Check Logs
```bash
tail -f runs/test/my_video_finetune_test/training.log
```

### Use Weights & Biases
```bash
# Add to your training command:
--report_to wandb
```

### Validate Loss Decrease
Training is working if:
- Loss decreases over steps
- No CUDA OOM errors
- Data loads without errors

## Troubleshooting

### Problem: OOM (Out of Memory)
**Solutions:**
```bash
--per_device_train_batch_size 1
--gradient_accumulation_steps 16
--gradient_checkpointing True
--model_max_length 1024  # Reduce sequence length
```

### Problem: Data Loading Errors
**Check:**
1. Video paths are correct and absolute
2. Videos are readable (test with `ffmpeg -i video.mp4`)
3. JSONL format is valid (test with `jq . train.jsonl`)

### Problem: Slow Training
**Optimize:**
```bash
--dataloader_num_workers 8  # Parallel data loading
--fp16 True                 # or --bf16 True
--gradient_checkpointing False  # Trade memory for speed
```

### Problem: Model Not Learning
**Try:**
1. Increase learning rate: `--learning_rate 5e-5`
2. Check dataset quality and labels
3. Increase training steps: `--max_steps 1000`
4. Verify conversations follow proper format

## Example: Adapting VILA for Custom Task

Here's how you adapted VILA for exercise analysis:

```jsonl
{
  "id": "vid_00012865",
  "video": "/data/exercises/high_knees/00012865.mp4",
  "num_frames": 10,
  "fps": 30,
  "conversations": [
    {
      "from": "user",
      "value": "You are a physiotherapy assistant. Analyze the video and return a JSON object..."
    },
    {
      "from": "assistant",
      "value": "{\"labels\": [\"high knees - speed=2.25 rps\", \"high knees - angle=45.0 degrees\"]}"
    }
  ]
}
```

**Key Considerations:**
1. ✅ Clear task instruction in user prompt
2. ✅ Structured output format (JSON)
3. ✅ Domain-specific terminology
4. ✅ Consistent num_frames for similar videos

## Next Steps

1. ✅ Test dataset loads: `./test_finetune.sh`
2. ⏳ Run full fine-tune: `bash scripts/NVILA-Lite/sft.sh ...`
3. ⏳ Evaluate model: `vila-eval --model-path runs/train/...`
4. ⏳ Deploy: Use `vila-infer` or `server.py`

## Resources

- [VILA Paper](https://arxiv.org/abs/2412.04468)
- [Training Scripts](scripts/NVILA-Lite/)
- [Evaluation Guide](README.md#evaluations)
- [Inference Guide](README.md#inference)

---

**Questions?** Check the main [README.md](README.md) or open an issue on GitHub.

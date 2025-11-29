# VILA Dataset Format Examples

This document shows the exact format for different types of datasets in VILA, with real examples from the codebase.

## 1. Single Image Q&A Dataset (JSON Format)

**Example from `finetuning/README.md`:**

```json
[
    {
        "id": "sample_001",
        "image": "images/1.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "What can you see in the image?"
            },
            {
                "from": "gpt",
                "value": "In the center of the image, I can see..."
            }
        ]
    },
    {
        "id": "sample_002",
        "image": "images/2.jpg",
        "conversations": [
            {
                "from": "human",
                "value": "How many chairs are there in the image?"
            },
            {
                "from": "gpt",
                "value": "There are totally 3 chairs..."
            }
        ]
    }
]
```

**Registry entry (default.yaml):**
```yaml
SampleQA:
    _target_: llava.data.LLaVADataset
    data_path: /path/to/dataset.json
    media_dir: /path/to/root/folder
```

**Folder structure:**
```
dataset/
├── dataset.json
└── images/
    ├── 1.jpg
    ├── 2.jpg
    └── ...
```

---

## 2. Video Dataset (JSON Format)

**Example from `finetuning/README.md`:**

```json
[
    {
        "id": "video_001",
        "video": "videos/1.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "Explain the video's components, including its characters, setting, and plot."
            },
            {
                "from": "gpt",
                "value": "The video sequence commences with a white screen that transitions..."
            }
        ]
    },
    {
        "id": "video_002",
        "video": "videos/2.mp4",
        "conversations": [
            {
                "from": "human",
                "value": "What is happening in this video?"
            },
            {
                "from": "gpt",
                "value": "This video shows..."
            }
        ]
    }
]
```

**Registry entry:**
```yaml
SampleVideo:
    _target_: llava.data.LLaVADataset
    data_path: /path/to/dataset.json
    media_dir: /path/to/root/folder
    is_video: true
```

**Folder structure:**
```
dataset/
├── dataset.json
└── videos/
    ├── 1.mp4
    ├── 2.mp4
    └── ...
```

---

## 3. Video Dataset (JSONL Format) - YOUR FORMAT

**Example (your actual data from `data/my_video_sft/train.jsonl`):**

```jsonl
{"id":"vid_00012865","video":"/data/exercises/high_knees/00012865.mp4","num_frames":10,"fps":30,"conversations":[{"from":"user","value":"You are a physiotherapy assistant. Analyze the video and return a JSON object with keys `labels` and `labels_descriptive` describing the exercise and parameters."},{"from":"assistant","value":"{\"labels\": [\"high knees - speed=2.25 rps\", \"high knees - angle=45.0 degrees\"], \"labels_descriptive\": [\"high knees - speed=2.25 rps\", \"high knees - angle=45.0 degrees\"]}"}]}
{"id":"vid_00006266","video":"/data/exercises/high_knees_march/00006266.mp4","num_frames":10,"conversations":[{"from":"user","value":"You are a physiotherapy assistant. Analyze the video and return a JSON object with keys `labels` and `labels_descriptive` describing the exercise and parameters."},{"from":"assistant","value":"{\"labels\": [\"high knees march - no issues\", \"high knees march - speed=0.60 rps\", \"high knees march - height=2\"], \"labels_descriptive\": [\"high knees march - no obvious issue\", \"high knees march - speed=0.60 rps\", \"high knees march - height=2\"]}"}]}
```

**Prettified version:**
```json
{
    "id": "vid_00012865",
    "video": "/data/exercises/high_knees/00012865.mp4",
    "num_frames": 10,
    "fps": 30,
    "conversations": [
        {
            "from": "user",
            "value": "You are a physiotherapy assistant. Analyze the video..."
        },
        {
            "from": "assistant",
            "value": "{\"labels\": [\"high knees - speed=2.25 rps\"]}"
        }
    ]
}
```

**Registry entry (your `llava/data/registry/datasets/my_video.yaml`):**
```yaml
my_video_sft:
  type: jsonl_conversation_video
  ann_path: /data/my_video_sft/train.jsonl
  media_dir: /
```

---

## 4. Multi-Image Dataset

**Example:**

```json
[
    {
        "id": "multi_001",
        "image": ["images/1.jpg", "images/2.jpg", "images/3.jpg"],
        "conversations": [
            {
                "from": "human",
                "value": "Compare these three images."
            },
            {
                "from": "gpt",
                "value": "The first image shows..., the second image depicts..., and the third..."
            }
        ]
    }
]
```

---

## 5. OCR/Document Understanding Dataset

**Example from `data_prepare/sft/SROIE.py`:**

```jsonl
{"id": "receipt_001", "image": "IMG_001.jpg", "conversations": [{"from": "human", "value": "<image>\nwhat is the name of the company that issued this receipt? Answer this question using the text in the image directly."}, {"from": "gpt", "value": "ABC Company Pte Ltd"}]}
{"id": "receipt_002", "image": "IMG_002.jpg", "conversations": [{"from": "human", "value": "<image>\nwhat is the total amount of this receipt? Answer this question using the text in the image directly."}, {"from": "gpt", "value": "$42.50"}]}
```

---

## 6. Multi-Turn Conversation Dataset

**Example from `data_prepare/sft/preprocess_metamathqa.py`:**

```json
[
    {
        "id": 0,
        "dataset_name": "metamathqa",
        "conversations": [
            {
                "from": "human",
                "value": "What is 2 + 2?"
            },
            {
                "from": "gpt",
                "value": "2 + 2 = 4"
            },
            {
                "from": "human",
                "value": "What about 4 + 4?"
            },
            {
                "from": "gpt",
                "value": "4 + 4 = 8"
            }
        ]
    }
]
```

---

## 7. Dense Video Captioning (with timestamps)

**Example from `llava/data/dataset_impl/lita.py` (DVCDataset):**

```json
{
    "id": "video_001",
    "video_path": "/path/to/video.mp4",
    "duration": 120.5,
    "timestamps": [[0.0, 5.2], [5.2, 12.8], [12.8, 20.5]],
    "sentences": [
        "A person walks into the room",
        "They pick up a book from the table",
        "They sit down and begin reading"
    ]
}
```

**Note:** This format auto-generates questions like "Provide a detailed description of the given video. Each sentence should begin with the start and end timestamps."

---

## Key Field Explanations

### Required Fields:

- **`id`**: Unique identifier (string or number)
- **`image`** OR **`video`**: Path to media file (relative to `media_dir`)
  - Can be a string for single file
  - Can be a list for multiple files
- **`conversations`**: List of conversation turns
  - Each turn has `"from"` (human/user/gpt/assistant) and `"value"` (text content)

### Optional Fields:

- **`num_frames`**: Override number of frames to extract from video
- **`fps`**: Override frame rate for video sampling
- **`dataset_name`**: Source dataset identifier
- **`datasource`**: Data source metadata

### Special Tokens:

- **`<image>`**: Automatically added if not present (usually at the beginning)
- **`<video>`**: Not explicitly needed for video datasets
- You DON'T need to add these tokens manually unless using specific formats

---

## Conversation Role Names

VILA accepts multiple formats for conversation roles:

| Human/User | Assistant/GPT |
|------------|---------------|
| `"human"` | `"gpt"` |
| `"user"` | `"assistant"` |

Both formats work identically.

---

## File Format: JSON vs JSONL

### JSON Format:
```json
[
    {"id": 1, "image": "1.jpg", "conversations": [...]},
    {"id": 2, "image": "2.jpg", "conversations": [...]}
]
```

### JSONL Format (one JSON per line):
```jsonl
{"id": 1, "image": "1.jpg", "conversations": [...]}
{"id": 2, "image": "2.jpg", "conversations": [...]}
```

**Both formats are supported!** JSONL is better for:
- Large datasets (can be streamed)
- Incremental processing
- Easier to append new data

---

## Real Examples from VILA Datasets

### 1. DummyDataset (for testing)
```python
{
    "id": 0,
    "image": "imagenet_cat.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat is in the image?"},
        {"from": "gpt", "value": "A cat."}
    ]
}
```

### 2. DocReason Dataset
```jsonl
{"id": 0, "image": "doc_001.png", "conversations": [{"from": "human", "value": "<image>\nWhat is the main topic of this document?"}, {"from": "gpt", "value": "The document discusses..."}]}
```

### 3. idefics2 Dataset (processed)
```json
{
    "id": 1234,
    "image": "images/dataset_name/1234.png",
    "conversations": [
        {"from": "human", "value": "Describe the image"},
        {"from": "gpt", "value": "The image shows..."}
    ]
}
```

---

## Creating Your Own Dataset

### Step 1: Prepare your data in one of these formats

**Option A - Simple JSON:**
```python
import json

data = [
    {
        "id": "sample_001",
        "video": "videos/sample_001.mp4",
        "conversations": [
            {"from": "human", "value": "What is happening?"},
            {"from": "gpt", "value": "Someone is exercising."}
        ]
    }
]

with open("dataset.json", "w") as f:
    json.dump(data, f, indent=2)
```

**Option B - JSONL (better for large datasets):**
```python
import json

samples = [
    {"id": "001", "video": "v001.mp4", "conversations": [...]},
    {"id": "002", "video": "v002.mp4", "conversations": [...]},
]

with open("dataset.jsonl", "w") as f:
    for sample in samples:
        f.write(json.dumps(sample) + "\n")
```

### Step 2: Register in `llava/data/registry/datasets/default.yaml`

```yaml
my_custom_dataset:
    _target_: llava.data.LLaVADataset
    data_path: /path/to/dataset.json  # or .jsonl
    media_dir: /path/to/media/folder
    is_video: true  # Set to true for video datasets
```

### Step 3: Use in training

```bash
bash scripts/NVILA-Lite/sft.sh <model_path> my_custom_dataset
```

---

## Validation Checklist

Before training, verify:

- ✅ All media files exist at specified paths
- ✅ Paths are either absolute or relative to `media_dir`
- ✅ JSON/JSONL is valid (test with `jq` or `json.loads()`)
- ✅ Every sample has `id`, media field, and `conversations`
- ✅ Conversations alternate between human/gpt roles
- ✅ Videos are readable (test with `ffmpeg -i video.mp4`)
- ✅ Dataset is registered in YAML file

---

## Quick Test Script

```python
import json

# For JSON
with open("dataset.json") as f:
    data = json.load(f)
    print(f"Loaded {len(data)} samples")
    print("First sample:", data[0])

# For JSONL
with open("dataset.jsonl") as f:
    lines = f.readlines()
    print(f"Loaded {len(lines)} samples")
    first = json.loads(lines[0])
    print("First sample:", first)
```

---

## Summary

Your dataset format (`train.jsonl`) follows the standard VILA video format perfectly! The key points:

1. ✅ Uses JSONL format (one JSON per line)
2. ✅ Has required fields: `id`, `video`, `conversations`
3. ✅ Conversations use `from: user/assistant` (standard format)
4. ✅ Properly registered in `my_video.yaml`
5. ✅ Ready to use with existing training scripts

You're all set to fine-tune! 🎉

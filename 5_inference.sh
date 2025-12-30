#!/bin/bash
set -e

# ==================================================
# Paths
# ==================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/llava/data/registry/datasets/dataset"
INPUT_JSON="$DATASET_DIR/test.json"
OUTPUT_DIR="$DATASET_DIR/csv"

mkdir -p "$OUTPUT_DIR"

# ==================================================
# Models
# ==================================================
BASE_MODEL="Efficient-Large-Model/NVILA-Lite-2B"
FT_MODEL="Efficient-Large-Model/NVILA-Lite-2B"

# ==================================================
# Limit for testing
# ==================================================
LIMIT=5  # Only first 5 samples

# ==================================================
# Run inference for base model
# ==================================================
echo "✅ Running inference for BASE model..."
python "$SCRIPT_DIR/llava/inference/inference.py" \
    "$INPUT_JSON" \
    "$OUTPUT_DIR/base_model_inference.csv" \
    "$BASE_MODEL" \
    "$LIMIT"

# ==================================================
# Run inference for FT model
# ==================================================
echo "✅ Running inference for FT model..."
python "$SCRIPT_DIR/llava/inference/inference.py" \
    "$INPUT_JSON" \
    "$OUTPUT_DIR/ft_model_inference.csv" \
    "$FT_MODEL" \
    "$LIMIT"

echo "✅ Inference done. CSVs saved to $OUTPUT_DIR"

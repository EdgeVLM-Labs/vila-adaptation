#!/bin/bash

# Portable script directory detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/llava/data/registry/datasets/dataset"

PYTHON_SCRIPT="$SCRIPT_DIR/llava/inference/inference.py"

INPUT_JSON="$DATASET_DIR/test.json"
OUTPUT_CSV="$DATASET_DIR/csv/inference_results.csv"
MODEL_PATH="Efficient-Large-Model/VILA1.5-3b"

LIMIT=""  # Set to a number to limit, or leave empty for all

if [ -z "$LIMIT" ]; then
    python "$PYTHON_SCRIPT" "$INPUT_JSON" "$OUTPUT_CSV" "$MODEL_PATH"
else
    python "$PYTHON_SCRIPT" "$INPUT_JSON" "$OUTPUT_CSV" "$MODEL_PATH" "$LIMIT"
fi
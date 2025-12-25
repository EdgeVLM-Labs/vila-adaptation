#!/bin/bash

# Portable script directory detection
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/llava/data/registry/datasets/dataset"

# Python script path
PYTHON_SCRIPT="$SCRIPT_DIR/llava/inference/generate_test_report.py"

# Input: predictions JSON file (output from your inference script)
# This should contain the model predictions you want to evaluate
PREDICTIONS_JSON="$DATASET_DIR/test_predictions.json"

# Output paths
OUTPUT_EXCEL="$DATASET_DIR/test_evaluation_report.xlsx"
OUTPUT_CSV="$DATASET_DIR/test_evaluation_report.csv"

# Optional flags
USE_BERT=true        # Set to false to skip BERT similarity (faster)
USE_LLM_JUDGE=false  # Set to true to enable LLM judge evaluation (slower, requires GPU)

echo "=================================================="
echo "VILA Model Evaluation Report Generation"
echo "=================================================="
echo "Script Directory: $SCRIPT_DIR"
echo "Dataset Directory: $DATASET_DIR"
echo "Predictions File: $PREDICTIONS_JSON"
echo "Output Excel: $OUTPUT_EXCEL"
echo "Output CSV: $OUTPUT_CSV"
echo "Use BERT: $USE_BERT"
echo "Use LLM Judge: $USE_LLM_JUDGE"
echo "=================================================="
echo ""

# Check if predictions file exists
if [ ! -f "$PREDICTIONS_JSON" ]; then
    echo "❌ Error: Predictions file not found at: $PREDICTIONS_JSON"
    echo ""
    echo "Please ensure you have run inference first to generate predictions."
    echo "The predictions file should be a JSON file containing:"
    echo "  - video_path"
    echo "  - ground_truth"
    echo "  - prediction"
    echo "  - status (optional)"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "❌ Error: Python script not found at: $PYTHON_SCRIPT"
    exit 1
fi

# Build command with flags
CMD="python \"$PYTHON_SCRIPT\" --predictions \"$PREDICTIONS_JSON\" --output \"$OUTPUT_EXCEL\" --csv-output \"$OUTPUT_CSV\""

if [ "$USE_BERT" = false ]; then
    CMD="$CMD --no-bert"
fi

if [ "$USE_LLM_JUDGE" = false ]; then
    CMD="$CMD --no-llm-judge"
fi

echo "Running command:"
echo "$CMD"
echo ""

# Execute the command
eval $CMD

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================="
    echo "✓ Report generation completed successfully!"
    echo "=================================================="
    echo "Generated files:"
    echo "  Excel: $OUTPUT_EXCEL"
    echo "  CSV: $OUTPUT_CSV"
    echo "=================================================="
else
    echo ""
    echo "=================================================="
    echo "❌ Report generation failed!"
    echo "=================================================="
    exit 1
fi
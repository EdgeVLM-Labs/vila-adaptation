#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_DIR="$SCRIPT_DIR/llava/data/registry/datasets/dataset"
CSV_DIR="$DATASET_DIR/csv"

BASE_CSV="$CSV_DIR/base_model_inference.csv"
FT_CSV="$CSV_DIR/ft_model_inference.csv"
REPORT_XLSX="$DATASET_DIR/test_evaluation_report.xlsx"

echo "✅ Generating test evaluation report..."

python "$SCRIPT_DIR/llava/inference/generate_test_report.py" \
    --predictions "$FT_CSV" \
    --base-predictions "$BASE_CSV" \
    --output "$REPORT_XLSX"

echo "✅ Report saved to $REPORT_XLSX"

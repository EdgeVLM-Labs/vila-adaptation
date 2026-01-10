#!/bin/bash

# ============================================================================
# Evaluation Pipeline Runner
# ============================================================================
# This script runs the unified evaluation pipeline to:
#   1. Match videos between base and finetuned model results
#   2. Clean model outputs
#   3. Remove duplicate video entries
#   4. Calculate evaluation scores (BERT, METEOR, ROUGE)
#
# Usage:
#   bash 7_calc_scores_inf_results.sh <base_csv> <finetuned_csv> [output_dir] [--no-bert]
#
# Arguments:
#   base_csv        - Path to the base model inference CSV file (required)
#   finetuned_csv   - Path to the finetuned model inference CSV file (required)
#   output_dir      - Directory to save results (optional, default: ./output)
#   --no-bert       - Skip BERT calculations for faster execution (optional)
#
# Examples:
#   bash 7_calc_scores_inf_results.sh base_results.csv finetuned_results.csv
#   bash 7_calc_scores_inf_results.sh base.csv finetuned.csv ./my_results
#   bash 7_calc_scores_inf_results.sh base.csv finetuned.csv ./output --no-bert
#
# Required CSV Format (for both inputs):
#   - video: Video identifier
#   - ground_truth: Ground truth text
#   - model_output: Model prediction
# ============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if at least two CSV files are provided
if [ $# -lt 2 ]; then
    echo -e "${RED}Error: Two CSV files are required (base and finetuned)${NC}"
    echo ""
    echo "Usage: bash 7_calc_scores_inf_results.sh <base_csv> <finetuned_csv> [output_dir] [--no-bert]"
    echo ""
    echo "Examples:"
    echo "  bash 7_calc_scores_inf_results.sh base_results.csv finetuned_results.csv"
    echo "  bash 7_calc_scores_inf_results.sh base.csv finetuned.csv ./my_results"
    echo "  bash 7_calc_scores_inf_results.sh base.csv finetuned.csv ./output --no-bert"
    exit 1
fi

BASE_CSV="$1"
FINETUNED_CSV="$2"
OUTPUT_DIR="${3:-./output}"
NO_BERT_FLAG=""

# Check for --no-bert flag in positions 3 or 4
if [ "$4" = "--no-bert" ] || [ "$3" = "--no-bert" ]; then
    NO_BERT_FLAG="--no-bert"
    if [ "$3" = "--no-bert" ]; then
        OUTPUT_DIR="./output"
    fi
fi

# Check if input files exist
if [ ! -f "$BASE_CSV" ]; then
    echo -e "${RED}Error: Base CSV file not found: $BASE_CSV${NC}"
    exit 1
fi

if [ ! -f "$FINETUNED_CSV" ]; then
    echo -e "${RED}Error: Finetuned CSV file not found: $FINETUNED_CSV${NC}"
    exit 1
fi

echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}         EVALUATION PIPELINE RUNNER${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo -e "Base CSV:        ${YELLOW}$BASE_CSV${NC}"
echo -e "Finetuned CSV:   ${YELLOW}$FINETUNED_CSV${NC}"
echo -e "Output Dir:      ${YELLOW}$OUTPUT_DIR${NC}"
if [ -n "$NO_BERT_FLAG" ]; then
    echo -e "BERT:            ${YELLOW}Disabled (faster mode)${NC}"
else
    echo -e "BERT:            ${YELLOW}Enabled${NC}"
fi
echo -e "${GREEN}============================================================================${NC}"
echo ""

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Run the Python pipeline
python3 "$SCRIPT_DIR/result_replication/evaluation_pipeline.py" \
    --base-csv "$BASE_CSV" \
    --finetuned-csv "$FINETUNED_CSV" \
    --output-dir "$OUTPUT_DIR" \
    $NO_BERT_FLAG

# Check if the pipeline succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "${GREEN}✓ Evaluation pipeline completed successfully!${NC}"
    echo -e "${GREEN}============================================================================${NC}"
    echo -e "Results saved in: ${YELLOW}$OUTPUT_DIR${NC}"
    echo ""
else
    echo ""
    echo -e "${RED}============================================================================${NC}"
    echo -e "${RED}✗ Evaluation pipeline failed!${NC}"
    echo -e "${RED}============================================================================${NC}"
    exit 1
fi

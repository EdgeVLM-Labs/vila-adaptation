#!/usr/bin/env python3
"""
Unified Evaluation Pipeline

This script processes inference results through four stages:
1. Match videos between base and finetuned model results (keep only common videos)
2. Clean model outputs (remove log lines)
3. Remove duplicate video entries
4. Calculate evaluation scores (BERT, METEOR, ROUGE)

Usage:
    python evaluation_pipeline.py --base-csv base_results.csv --finetuned-csv finetuned_results.csv --output-dir ./output
    
Or run via shell script:
    bash 7_calc_scores_inf_results.sh base_results.csv finetuned_results.csv ./output

Required CSV format for both inputs:
    - video: Video identifier
    - ground_truth: Ground truth text
    - model_output: Model prediction
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import evaluate
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from nltk.translate.meteor_score import meteor_score
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.legend import Legend
from openpyxl.chart.series import DataPoint
from openpyxl.chart.shapes import GraphicalProperties


# ============================================================================
# CONFIGURATION
# ============================================================================
USE_BERT = True  # Set to False to skip BERT (faster)

# Score thresholds (0-1 scale)
ROUGE_GREEN_THRESHOLD = 0.5
ROUGE_YELLOW_THRESHOLD = 0.2
BERT_GREEN_THRESHOLD = 0.7
BERT_YELLOW_THRESHOLD = 0.4
METEOR_GREEN_THRESHOLD = 0.5
METEOR_YELLOW_THRESHOLD = 0.2


# ============================================================================
# STAGE 1: MATCH VIDEOS BETWEEN BASE AND FINETUNED
# ============================================================================
def match_videos(base_df: pd.DataFrame, finetuned_df: pd.DataFrame) -> pd.DataFrame:
    """Match videos between base and finetuned results, keep only common videos."""
    print("\n" + "="*80)
    print("STAGE 1: MATCHING VIDEOS BETWEEN BASE AND FINETUNED MODELS")
    print("="*80)
    
    print(f"Base model CSV: {len(base_df)} rows")
    print(f"Finetuned model CSV: {len(finetuned_df)} rows")
    
    # Get common videos
    base_videos = set(base_df['video'].values)
    finetuned_videos = set(finetuned_df['video'].values)
    common_videos = base_videos.intersection(finetuned_videos)
    
    print(f"\nCommon videos: {len(common_videos)}")
    print(f"Base-only videos: {len(base_videos - finetuned_videos)}")
    print(f"Finetuned-only videos: {len(finetuned_videos - base_videos)}")
    
    if len(common_videos) == 0:
        raise ValueError("No common videos found between base and finetuned CSVs!")
    
    # Filter both dataframes to keep only common videos
    base_filtered = base_df[base_df['video'].isin(common_videos)].copy()
    finetuned_filtered = finetuned_df[finetuned_df['video'].isin(common_videos)].copy()
    
    # Sort by video to ensure alignment
    base_filtered = base_filtered.sort_values('video').reset_index(drop=True)
    finetuned_filtered = finetuned_filtered.sort_values('video').reset_index(drop=True)
    
    # Merge the dataframes
    merged_df = pd.merge(
        base_filtered[['video', 'ground_truth', 'model_output']],
        finetuned_filtered[['video', 'model_output']],
        on='video',
        how='inner',
        suffixes=('_base', '_finetuned')
    )
    
    # Rename columns for clarity
    merged_df.rename(columns={
        'model_output_base': 'base_model_output',
        'model_output_finetuned': 'finetuned_model_output'
    }, inplace=True)
    
    print(f"\n✓ Merged dataset: {len(merged_df)} rows")
    
    return merged_df


# ============================================================================
# STAGE 2: CLEAN MODEL OUTPUTS
# ============================================================================
def clean_model_output(text):
    """Remove log lines from model output."""
    if not isinstance(text, str):
        return text

    lines = text.splitlines()
    cleaned_lines = [
        line for line in lines
        if "Setting ds_accelerator" not in line
    ]
    return "\n".join(cleaned_lines).strip()


def clean_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Clean model outputs in dataframe."""
    print("\n" + "="*80)
    print("STAGE 2: CLEANING MODEL OUTPUTS")
    print("="*80)
    
    # Clean both model outputs
    if 'base_model_output' in df.columns:
        df['base_model_output'] = df['base_model_output'].apply(clean_model_output)
        print("✓ Cleaned base_model_output")
    
    if 'finetuned_model_output' in df.columns:
        df['finetuned_model_output'] = df['finetuned_model_output'].apply(clean_model_output)
        print("✓ Cleaned finetuned_model_output")
    
    # Also clean model_output if it exists (for single model case)
    if 'model_output' in df.columns:
        df['model_output'] = df['model_output'].apply(clean_model_output)
        print("✓ Cleaned model_output")
    
    return df


# ============================================================================
# STAGE 3: REMOVE DUPLICATES
# ============================================================================
def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate video entries."""
    print("\n" + "="*80)
    print("STAGE 3: REMOVING DUPLICATE ENTRIES")
    print("="*80)
    
    original_count = len(df)
    
    # Check for duplicates
    duplicate_mask = df.duplicated(subset=["video"], keep=False)
    duplicates = df[duplicate_mask]
    
    if not duplicates.empty:
        print(f"⚠ Found {len(duplicates)} duplicate entries:")
        print(duplicates["video"].value_counts())
    else:
        print("✓ No duplicates found")
    
    # Remove duplicates (keep first occurrence)
    df_cleaned = df.drop_duplicates(subset=["video"], keep="first")
    
    print(f"Rows: {original_count} → {len(df_cleaned)} (removed {original_count - len(df_cleaned)})")
    
    return df_cleaned


# ============================================================================
# STAGE 4: CALCULATE EVALUATION SCORES
# ============================================================================
def compute_meteor_score(reference: str, hypothesis: str) -> float:
    if not reference or not hypothesis:
        return 0.0
    try:
        return meteor_score([reference.split()], hypothesis.split())
    except Exception:
        return 0.0


def compute_rouge_score(reference: str, hypothesis: str, metric) -> float:
    """Compute ROUGE-L score."""
    if not reference or not hypothesis or metric is None:
        return 0.0
    try:
        result = metric.compute(predictions=[hypothesis], references=[reference])
        return result['rougeL']
    except:
        return 0.0


def extract_exercise_name(text: str) -> str:
    """Extract exercise name from text (before the dash)."""
    if not text:
        return ""
    if '-' in text:
        return text.split('-')[0].strip().lower()
    return text.strip().lower()


def check_exercise_match(ground_truth: str, prediction: str) -> bool:
    """Check if exercise names match."""
    gt_exercise = extract_exercise_name(ground_truth)
    pred_exercise = extract_exercise_name(prediction)
    if not gt_exercise or not pred_exercise:
        return False
    return gt_exercise == pred_exercise


def compute_cosine_similarity_bert(text1: str, text2: str, model) -> float:
    """Compute cosine similarity using BERT embeddings."""
    if not text1 or not text2:
        return 0.0
    try:
        embeddings = model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        return float(similarity)
    except:
        return 0.0


def calculate_scores(df: pd.DataFrame, use_bert: bool = True):
    """Calculate evaluation scores for both models."""
    print("\n" + "="*80)
    print("STAGE 4: CALCULATING EVALUATION SCORES")
    print("="*80)
    
    # Load models and metrics
    bert_model = None
    if use_bert:
        print("\nLoading BERT model...")
        try:
            bert_model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ BERT model loaded")
        except Exception as e:
            print(f"⚠ Failed to load BERT model: {e}")
            use_bert = False
    
    print("Loading evaluation metrics...")
    meteor_metric = None
    rouge_metric = None
    
    try:
        meteor_metric = evaluate.load('meteor')
        print("✓ METEOR metric loaded")
    except Exception as e:
        print(f"⚠ Failed to load METEOR metric: {e}")
    
    try:
        rouge_metric = evaluate.load('rouge')
        print("✓ ROUGE metric loaded")
    except Exception as e:
        print(f"⚠ Failed to load ROUGE metric: {e}")
    
    # Calculate scores for base model
    print("\n" + "-"*80)
    print("Calculating scores for BASE MODEL...")
    print("-"*80)
    
    base_bert_scores = []
    base_meteor_scores = []
    base_rouge_scores = []
    base_exercise_matches = []
    
    for idx, row in df.iterrows():
        gt = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ''
        pred = str(row['base_model_output']) if pd.notna(row['base_model_output']) else ''
        
        if use_bert and bert_model:
            bert_sim = compute_cosine_similarity_bert(gt, pred, bert_model)
            base_bert_scores.append(bert_sim)
        
        meteor_sim = compute_meteor_score(gt, pred)
        base_meteor_scores.append(meteor_sim)
        
        rouge_sim = compute_rouge_score(gt, pred, rouge_metric)
        base_rouge_scores.append(rouge_sim)
        
        exercise_match = check_exercise_match(gt, pred)
        base_exercise_matches.append(exercise_match)
    
    # Add base model scores to dataframe
    if use_bert and base_bert_scores:
        df['base_bert_similarity'] = base_bert_scores
    df['base_meteor_score'] = base_meteor_scores
    df['base_rouge_score'] = base_rouge_scores
    df['base_exercise_match'] = base_exercise_matches
    
    # Calculate scores for finetuned model
    print("\n" + "-"*80)
    print("Calculating scores for FINETUNED MODEL...")
    print("-"*80)
    
    finetuned_bert_scores = []
    finetuned_meteor_scores = []
    finetuned_rouge_scores = []
    finetuned_exercise_matches = []
    
    for idx, row in df.iterrows():
        gt = str(row['ground_truth']) if pd.notna(row['ground_truth']) else ''
        pred = str(row['finetuned_model_output']) if pd.notna(row['finetuned_model_output']) else ''
        
        if use_bert and bert_model:
            bert_sim = compute_cosine_similarity_bert(gt, pred, bert_model)
            finetuned_bert_scores.append(bert_sim)
        
        meteor_sim = compute_meteor_score(gt, pred)
        finetuned_meteor_scores.append(meteor_sim)
        
        rouge_sim = compute_rouge_score(gt, pred, rouge_metric)
        finetuned_rouge_scores.append(rouge_sim)
        
        exercise_match = check_exercise_match(gt, pred)
        finetuned_exercise_matches.append(exercise_match)
    
    # Add finetuned model scores to dataframe
    if use_bert and finetuned_bert_scores:
        df['finetuned_bert_similarity'] = finetuned_bert_scores
    df['finetuned_meteor_score'] = finetuned_meteor_scores
    df['finetuned_rouge_score'] = finetuned_rouge_scores
    df['finetuned_exercise_match'] = finetuned_exercise_matches
    
    scores = {
        'base': {
            'bert': base_bert_scores if use_bert else None,
            'meteor': base_meteor_scores,
            'rouge': base_rouge_scores,
            'exercise': base_exercise_matches
        },
        'finetuned': {
            'bert': finetuned_bert_scores if use_bert else None,
            'meteor': finetuned_meteor_scores,
            'rouge': finetuned_rouge_scores,
            'exercise': finetuned_exercise_matches
        }
    }
    
    return df, scores, use_bert


# ============================================================================
# EXCEL REPORT GENERATION
# ============================================================================
def create_excel_report(df: pd.DataFrame, scores: dict, output_path: str, use_bert: bool):
    """Create Excel report with charts comparing both models."""
    
    wb = openpyxl.Workbook()
    
    # SUMMARY SHEET
    ws = wb.active
    ws.title = "Summary"
    
    # Define styles
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
    bold_font = Font(bold=True)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Title
    ws['A1'] = "MODEL COMPARISON SUMMARY"
    ws['A1'].font = Font(bold=True, size=14)
    ws.merge_cells('A1:D1')
    
    row = 3
    
    # Base Model Stats
    ws[f'A{row}'] = "BASE MODEL"
    ws[f'A{row}'].font = header_font
    ws[f'A{row}'].fill = header_fill
    ws.merge_cells(f'A{row}:D{row}')
    row += 1
    
    base_scores = scores['base']
    
    if use_bert and base_scores['bert']:
        ws[f'A{row}'] = "BERT Similarity"
        ws[f'B{row}'] = f"Mean: {np.mean(base_scores['bert']):.4f}"
        ws[f'C{row}'] = f"Median: {np.median(base_scores['bert']):.4f}"
        ws[f'D{row}'] = f"Std: {np.std(base_scores['bert']):.4f}"
        row += 1
    
    ws[f'A{row}'] = "METEOR Score"
    ws[f'B{row}'] = f"Mean: {np.mean(base_scores['meteor']):.4f}"
    ws[f'C{row}'] = f"Median: {np.median(base_scores['meteor']):.4f}"
    ws[f'D{row}'] = f"Std: {np.std(base_scores['meteor']):.4f}"
    row += 1
    
    ws[f'A{row}'] = "ROUGE-L Score"
    ws[f'B{row}'] = f"Mean: {np.mean(base_scores['rouge']):.4f}"
    ws[f'C{row}'] = f"Median: {np.median(base_scores['rouge']):.4f}"
    ws[f'D{row}'] = f"Std: {np.std(base_scores['rouge']):.4f}"
    row += 1
    
    exercise_correct = sum(base_scores['exercise'])
    exercise_total = len(base_scores['exercise'])
    exercise_acc = (exercise_correct / exercise_total * 100) if exercise_total > 0 else 0
    ws[f'A{row}'] = "Exercise Match"
    ws[f'B{row}'] = f"Correct: {exercise_correct}/{exercise_total}"
    ws[f'C{row}'] = f"Accuracy: {exercise_acc:.2f}%"
    row += 2
    
    # Finetuned Model Stats
    ws[f'A{row}'] = "FINETUNED MODEL"
    ws[f'A{row}'].font = header_font
    ws[f'A{row}'].fill = PatternFill(start_color="28A745", end_color="28A745", fill_type="solid")
    ws.merge_cells(f'A{row}:D{row}')
    row += 1
    
    finetuned_scores = scores['finetuned']
    
    if use_bert and finetuned_scores['bert']:
        ws[f'A{row}'] = "BERT Similarity"
        ws[f'B{row}'] = f"Mean: {np.mean(finetuned_scores['bert']):.4f}"
        ws[f'C{row}'] = f"Median: {np.median(finetuned_scores['bert']):.4f}"
        ws[f'D{row}'] = f"Std: {np.std(finetuned_scores['bert']):.4f}"
        row += 1
    
    ws[f'A{row}'] = "METEOR Score"
    ws[f'B{row}'] = f"Mean: {np.mean(finetuned_scores['meteor']):.4f}"
    ws[f'C{row}'] = f"Median: {np.median(finetuned_scores['meteor']):.4f}"
    ws[f'D{row}'] = f"Std: {np.std(finetuned_scores['meteor']):.4f}"
    row += 1
    
    ws[f'A{row}'] = "ROUGE-L Score"
    ws[f'B{row}'] = f"Mean: {np.mean(finetuned_scores['rouge']):.4f}"
    ws[f'C{row}'] = f"Median: {np.median(finetuned_scores['rouge']):.4f}"
    ws[f'D{row}'] = f"Std: {np.std(finetuned_scores['rouge']):.4f}"
    row += 1
    
    exercise_correct = sum(finetuned_scores['exercise'])
    exercise_total = len(finetuned_scores['exercise'])
    exercise_acc = (exercise_correct / exercise_total * 100) if exercise_total > 0 else 0
    ws[f'A{row}'] = "Exercise Match"
    ws[f'B{row}'] = f"Correct: {exercise_correct}/{exercise_total}"
    ws[f'C{row}'] = f"Accuracy: {exercise_acc:.2f}%"
    
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = 25
    ws.column_dimensions['C'].width = 25
    ws.column_dimensions['D'].width = 20
    
    # DETAILED RESULTS SHEET
    ws_detail = wb.create_sheet("Detailed Results")
    
    # Write the dataframe to Excel
    for r_idx, row in enumerate(df.itertuples(index=False), start=1):
        for c_idx, value in enumerate(row, start=1):
            ws_detail.cell(row=r_idx, column=c_idx, value=value)
    
    # Format header
    for col in range(1, len(df.columns) + 1):
        cell = ws_detail.cell(row=1, column=col)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
    
    ws_detail.freeze_panes = "A2"
    
    # Save workbook
    wb.save(output_path)
    print(f"✓ Excel report saved to: {output_path}")


# ============================================================================
# SUMMARY PRINTING
# ============================================================================
def print_summary(scores: dict, use_bert: bool):
    """Print summary statistics."""
    
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    print("\n" + "-"*80)
    print("BASE MODEL")
    print("-"*80)
    base_scores = scores['base']
    
    if use_bert and base_scores['bert']:
        print(f"BERT Similarity:  Mean={np.mean(base_scores['bert']):.4f}  "
              f"Median={np.median(base_scores['bert']):.4f}  "
              f"Std={np.std(base_scores['bert']):.4f}")
    
    print(f"METEOR Score:     Mean={np.mean(base_scores['meteor']):.4f}  "
          f"Median={np.median(base_scores['meteor']):.4f}  "
          f"Std={np.std(base_scores['meteor']):.4f}")
    
    print(f"ROUGE-L Score:    Mean={np.mean(base_scores['rouge']):.4f}  "
          f"Median={np.median(base_scores['rouge']):.4f}  "
          f"Std={np.std(base_scores['rouge']):.4f}")
    
    exercise_correct = sum(base_scores['exercise'])
    exercise_total = len(base_scores['exercise'])
    exercise_acc = (exercise_correct / exercise_total * 100) if exercise_total > 0 else 0
    print(f"Exercise Match:   {exercise_correct}/{exercise_total} ({exercise_acc:.2f}%)")
    
    print("\n" + "-"*80)
    print("FINETUNED MODEL")
    print("-"*80)
    finetuned_scores = scores['finetuned']
    
    if use_bert and finetuned_scores['bert']:
        print(f"BERT Similarity:  Mean={np.mean(finetuned_scores['bert']):.4f}  "
              f"Median={np.median(finetuned_scores['bert']):.4f}  "
              f"Std={np.std(finetuned_scores['bert']):.4f}")
    
    print(f"METEOR Score:     Mean={np.mean(finetuned_scores['meteor']):.4f}  "
          f"Median={np.median(finetuned_scores['meteor']):.4f}  "
          f"Std={np.std(finetuned_scores['meteor']):.4f}")
    
    print(f"ROUGE-L Score:    Mean={np.mean(finetuned_scores['rouge']):.4f}  "
          f"Median={np.median(finetuned_scores['rouge']):.4f}  "
          f"Std={np.std(finetuned_scores['rouge']):.4f}")
    
    exercise_correct = sum(finetuned_scores['exercise'])
    exercise_total = len(finetuned_scores['exercise'])
    exercise_acc = (exercise_correct / exercise_total * 100) if exercise_total > 0 else 0
    print(f"Exercise Match:   {exercise_correct}/{exercise_total} ({exercise_acc:.2f}%)")
    
    print("\n" + "="*80)
    print("IMPROVEMENT (Finetuned vs Base)")
    print("="*80)
    
    if use_bert and base_scores['bert'] and finetuned_scores['bert']:
        bert_improvement = np.mean(finetuned_scores['bert']) - np.mean(base_scores['bert'])
        print(f"BERT Similarity:  {bert_improvement:+.4f} ({bert_improvement/np.mean(base_scores['bert'])*100:+.2f}%)")
    
    meteor_improvement = np.mean(finetuned_scores['meteor']) - np.mean(base_scores['meteor'])
    print(f"METEOR Score:     {meteor_improvement:+.4f} ({meteor_improvement/np.mean(base_scores['meteor'])*100:+.2f}%)")
    
    rouge_improvement = np.mean(finetuned_scores['rouge']) - np.mean(base_scores['rouge'])
    print(f"ROUGE-L Score:    {rouge_improvement:+.4f} ({rouge_improvement/np.mean(base_scores['rouge'])*100:+.2f}%)")
    
    base_exercise_acc = (sum(base_scores['exercise']) / len(base_scores['exercise']) * 100)
    finetuned_exercise_acc = (sum(finetuned_scores['exercise']) / len(finetuned_scores['exercise']) * 100)
    exercise_improvement = finetuned_exercise_acc - base_exercise_acc
    print(f"Exercise Match:   {exercise_improvement:+.2f}%")
    
    print("="*80)


# ============================================================================
# MAIN PIPELINE
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Unified Evaluation Pipeline for Model Inference Results',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Required CSV Format (for both base and finetuned):
  Each input CSV must contain these columns:
    - video: Video identifier (string)
    - ground_truth: Ground truth text (string)
    - model_output: Model prediction (string)

The pipeline will:
  1. Match videos present in both CSVs (keeps only common videos)
  2. Clean model outputs (remove log lines)
  3. Remove duplicate video entries
  4. Calculate evaluation scores (BERT, METEOR, ROUGE-L, Exercise Match)

Examples:
  python evaluation_pipeline.py --base-csv base_results.csv --finetuned-csv finetuned_results.csv
  python evaluation_pipeline.py --base-csv base.csv --finetuned-csv finetuned.csv --output-dir ./output
  python evaluation_pipeline.py --base-csv base.csv --finetuned-csv finetuned.csv --no-bert
        """
    )
    
    parser.add_argument(
        '--base-csv',
        type=str,
        required=True,
        help='Path to base model inference CSV file'
    )
    
    parser.add_argument(
        '--finetuned-csv',
        type=str,
        required=True,
        help='Path to finetuned model inference CSV file'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save output files (default: ./output)'
    )
    
    parser.add_argument(
        '--no-bert',
        action='store_true',
        help='Skip BERT similarity calculation (faster)'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    base_path = Path(args.base_csv)
    finetuned_path = Path(args.finetuned_csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    csv_output_path = output_dir / "evaluation_results.csv"
    excel_output_path = output_dir / "evaluation_report.xlsx"
    
    print("="*80)
    print("UNIFIED EVALUATION PIPELINE")
    print("="*80)
    print(f"Base CSV:       {base_path}")
    print(f"Finetuned CSV:  {finetuned_path}")
    print(f"Output Dir:     {output_dir}")
    print(f"Use BERT:       {not args.no_bert}")
    print("="*80)
    
    # Validate input files
    if not base_path.exists():
        print(f"\n❌ ERROR: Base CSV not found: {base_path}")
        return 1
    
    if not finetuned_path.exists():
        print(f"\n❌ ERROR: Finetuned CSV not found: {finetuned_path}")
        return 1
    
    # Load CSVs
    print("\nLoading input CSVs...")
    try:
        base_df = pd.read_csv(base_path)
        print(f"✓ Loaded base CSV: {len(base_df)} rows")
    except Exception as e:
        print(f"❌ ERROR loading base CSV: {e}")
        return 1
    
    try:
        finetuned_df = pd.read_csv(finetuned_path)
        print(f"✓ Loaded finetuned CSV: {len(finetuned_df)} rows")
    except Exception as e:
        print(f"❌ ERROR loading finetuned CSV: {e}")
        return 1
    
    # Check required columns
    required_cols = ['video', 'ground_truth', 'model_output']
    
    base_missing = [col for col in required_cols if col not in base_df.columns]
    if base_missing:
        print(f"\n❌ ERROR: Base CSV missing required columns: {base_missing}")
        print(f"Required: {required_cols}")
        print(f"Found: {list(base_df.columns)}")
        return 1
    
    finetuned_missing = [col for col in required_cols if col not in finetuned_df.columns]
    if finetuned_missing:
        print(f"\n❌ ERROR: Finetuned CSV missing required columns: {finetuned_missing}")
        print(f"Required: {required_cols}")
        print(f"Found: {list(finetuned_df.columns)}")
        return 1
    
    # Stage 1: Match videos
    try:
        df = match_videos(base_df, finetuned_df)
    except Exception as e:
        print(f"\n❌ ERROR in matching: {e}")
        return 1
    
    # Stage 2: Clean
    df = clean_csv(df)
    
    # Stage 3: Remove duplicates
    df = remove_duplicates(df)
    
    # Stage 4: Calculate scores
    df_with_scores, scores, use_bert = calculate_scores(df, use_bert=not args.no_bert)
    
    # Save outputs
    print("\n" + "="*80)
    print("SAVING OUTPUTS")
    print("="*80)
    
    print(f"Saving CSV to: {csv_output_path}")
    df_with_scores.to_csv(csv_output_path, index=False)
    print(f"✓ CSV saved")
    
    print(f"Saving Excel report to: {excel_output_path}")
    create_excel_report(df_with_scores, scores, str(excel_output_path), use_bert)
    
    # Print summary
    print_summary(scores, use_bert)
    
    print(f"\n{'='*80}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"📊 Excel Report: {excel_output_path}")
    print(f"📄 CSV Results:  {csv_output_path}")
    print(f"{'='*80}")
    
    return 0


if __name__ == "__main__":
    exit(main())

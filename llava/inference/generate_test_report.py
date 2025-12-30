#!/usr/bin/env python3
"""
Generate Test Evaluation Report for NVILA/VILA Models

This script processes CSV inference results and generates comprehensive Excel reports
with multiple similarity metrics including BERT, METEOR, ROUGE-L, and optional LLM judging.

Usage:
    python generate_test_report.py --predictions inference_results.csv
    python generate_test_report.py --predictions inference_results.csv --output report.xlsx
    python generate_test_report.py --predictions inference_results.csv --base-predictions base_results.csv
"""

import argparse
import csv
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.legend import Legend
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from sentence_transformers import SentenceTransformer
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Metric thresholds
ROUGE_GREEN_THRESHOLD = 0.5
ROUGE_YELLOW_THRESHOLD = 0.2

BERT_GREEN_THRESHOLD = 0.7
BERT_YELLOW_THRESHOLD = 0.4

METEOR_GREEN_THRESHOLD = 0.5
METEOR_YELLOW_THRESHOLD = 0.2


class MetricsCalculator:
    """Handles all metric calculations."""
    
    def __init__(self, use_bert: bool = True):
        self.use_bert = use_bert
        
        # Load models
        self.bert_model = self._load_bert_model() if use_bert else None
        self.meteor_metric = self._load_metric('meteor')
        self.rouge_metric = self._load_metric('rouge')
    
    def _load_bert_model(self) -> Optional[SentenceTransformer]:
        """Load BERT model for semantic similarity."""
        print("Loading BERT model for semantic similarity...")
        try:
            model = SentenceTransformer('all-MiniLM-L6-v2')
            print("✓ BERT model loaded")
            return model
        except Exception as e:
            print(f"⚠ Failed to load BERT model: {e}")
            return None
    
    def _load_metric(self, metric_name: str):
        """Load evaluation metric."""
        try:
            metric = evaluate.load(metric_name)
            print(f"✓ {metric_name.upper()} metric loaded")
            return metric
        except Exception as e:
            print(f"⚠ Failed to load {metric_name} metric: {e}")
            return None
    
    def compute_bert_similarity(self, text1: str, text2: str) -> float:
        """Compute BERT cosine similarity."""
        if not text1 or not text2 or self.bert_model is None:
            return 0.0
        
        try:
            embeddings = self.bert_model.encode([text1, text2])
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            return float(similarity)
        except:
            return 0.0
    
    def compute_meteor(self, reference: str, hypothesis: str) -> float:
        """Compute METEOR score."""
        if not reference or not hypothesis or self.meteor_metric is None:
            return 0.0
        
        try:
            return self.meteor_metric.compute(
                predictions=[hypothesis], 
                references=[reference]
            )['meteor']
        except:
            return 0.0
    
    def compute_rouge(self, reference: str, hypothesis: str) -> float:
        """Compute ROUGE-L score."""
        if not reference or not hypothesis or self.rouge_metric is None:
            return 0.0
        
        try:
            result = self.rouge_metric.compute(
                predictions=[hypothesis], 
                references=[reference]
            )
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
    """Check if exercise names match between ground truth and prediction."""
    gt_exercise = extract_exercise_name(ground_truth)
    pred_exercise = extract_exercise_name(prediction)
    
    if not gt_exercise or not pred_exercise:
        return False
    
    return gt_exercise == pred_exercise


def load_csv_predictions(csv_path: str) -> List[Dict]:
    """Load predictions from CSV file."""
    results = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            results.append({
                'video_path': row.get('video', ''),
                'ground_truth': row.get('ground_truth', ''),
                'prediction': row.get('model_output', ''),
                'memory_usage_mb': float(row.get('memory_usage_mb', 0)),
                'inference_time_sec': float(row.get('inference_time_sec', 0)),
                'status': 'success' if row.get('model_output', '').strip() and not row.get('model_output', '').startswith('ERROR') else 'error',
                'error': row.get('model_output', '') if row.get('model_output', '').startswith('ERROR') else ''
            })
    
    return results


def apply_color_coding(cell, score: float, metric_type: str):
    """Apply color coding based on score and metric type."""
    thresholds = {
        'bert': (BERT_GREEN_THRESHOLD, BERT_YELLOW_THRESHOLD),
        'meteor': (METEOR_GREEN_THRESHOLD, METEOR_YELLOW_THRESHOLD),
        'rouge': (ROUGE_GREEN_THRESHOLD, ROUGE_YELLOW_THRESHOLD)
    }
    
    green_thresh, yellow_thresh = thresholds.get(metric_type, (0.5, 0.2))
    
    if score >= green_thresh:
        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    elif score >= yellow_thresh:
        cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    else:
        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")


def create_excel_report(results: List[Dict], output_path: str, calculator: MetricsCalculator,
                       base_predictions: List[Dict] = None):
    """Create comprehensive Excel report with all metrics and visualizations."""
    print(f"\nGenerating Excel report...")
    
    # Create base prediction lookup if provided
    base_pred_map = {}
    if base_predictions:
        for bp in base_predictions:
            base_pred_map[bp.get('video_path', '')] = bp.get('prediction', '')
        print(f"✓ Loaded {len(base_pred_map)} base model predictions")
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Test Evaluation Results"
    
    # Define styles
    header_font = Font(bold=True, color="FFFFFF", size=12)
    header_fill = PatternFill(start_color="2E86AB", end_color="2E86AB", fill_type="solid")
    header_alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell_alignment = Alignment(vertical="top", wrap_text=True)
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    # Build headers
    headers = ["ID", "Video Path", "Ground Truth", "Model Prediction"]
    
    # Add base model column if available
    if base_predictions:
        headers.append("Base Model Prediction")
    
    if calculator.use_bert:
        headers.append("BERT Similarity")
    
    headers.extend(["METEOR Score", "ROUGE-L Score", "Exercise Match", "Status"])
    
    # Write headers
    for col, header in enumerate(headers, start=1):
        cell = ws.cell(row=1, column=col)
        cell.value = header
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = header_alignment
        cell.border = border
    
    # Set column widths
    ws.column_dimensions['A'].width = 8
    ws.column_dimensions['B'].width = 40
    ws.column_dimensions['C'].width = 50
    ws.column_dimensions['D'].width = 50
    
    col_idx = 5
    if base_predictions:
        ws.column_dimensions[get_column_letter(col_idx)].width = 50
        col_idx += 1
    
    if calculator.use_bert:
        ws.column_dimensions[get_column_letter(col_idx)].width = 15
        col_idx += 1
    
    ws.column_dimensions[get_column_letter(col_idx)].width = 15
    ws.column_dimensions[get_column_letter(col_idx + 1)].width = 15
    ws.column_dimensions[get_column_letter(col_idx + 2)].width = 15
    ws.column_dimensions[get_column_letter(col_idx + 3)].width = 10
    
    ws.freeze_panes = "A2"
    
    # Process results
    print(f"Processing {len(results)} results...")
    
    bert_scores = []
    meteor_scores = []
    rouge_scores = []
    exercise_matches = []
    
    for idx, result in enumerate(results, start=1):
        row = idx + 1
        
        video_path = result.get('video_path', '')
        ground_truth = result.get('ground_truth', '')
        prediction = result.get('prediction', '')
        status = result.get('status', 'unknown')
        
        # Compute metrics
        bert_sim = None
        if calculator.use_bert:
            bert_sim = calculator.compute_bert_similarity(ground_truth, prediction)
            bert_scores.append(bert_sim)
        
        meteor_sim = calculator.compute_meteor(ground_truth, prediction)
        meteor_scores.append(meteor_sim)
        
        rouge_sim = calculator.compute_rouge(ground_truth, prediction)
        rouge_scores.append(rouge_sim)
        
        exercise_match = check_exercise_match(ground_truth, prediction)
        exercise_matches.append(exercise_match)
        
        # Write data
        col = 1
        ws.cell(row=row, column=col).value = idx
        col += 1
        ws.cell(row=row, column=col).value = video_path
        col += 1
        ws.cell(row=row, column=col).value = ground_truth
        col += 1
        ws.cell(row=row, column=col).value = prediction
        col += 1
        
        # Add base model response if available
        if base_predictions:
            base_pred = base_pred_map.get(video_path, 'N/A')
            ws.cell(row=row, column=col).value = base_pred
            col += 1
        
        # Track column indices for color coding
        bert_col_idx = None
        if calculator.use_bert and bert_sim is not None:
            ws.cell(row=row, column=col).value = round(bert_sim, 4)
            bert_col_idx = col
            col += 1
        
        meteor_col_idx = col
        ws.cell(row=row, column=col).value = round(meteor_sim, 4)
        col += 1
        
        rouge_col_idx = col
        ws.cell(row=row, column=col).value = round(rouge_sim, 4)
        col += 1
        
        exercise_col_idx = col
        ws.cell(row=row, column=col).value = "TRUE" if exercise_match else "FALSE"
        col += 1
        
        ws.cell(row=row, column=col).value = status
        col += 1
        
        # Apply formatting and color coding
        for c in range(1, col):
            cell = ws.cell(row=row, column=c)
            cell.alignment = cell_alignment
            cell.border = border
            
            if bert_col_idx and c == bert_col_idx:
                apply_color_coding(cell, bert_sim, 'bert')
            elif c == meteor_col_idx:
                apply_color_coding(cell, meteor_sim, 'meteor')
            elif c == rouge_col_idx:
                apply_color_coding(cell, rouge_sim, 'rouge')
            elif c == exercise_col_idx:
                if exercise_match:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # Create summary sheet
    create_summary_sheet(wb, results, bert_scores, meteor_scores, rouge_scores, 
                        exercise_matches, calculator)
    
    # Save workbook
    wb.save(output_path)
    print(f"✓ Excel report saved to: {output_path}")
    
    # Print summary
    print_summary(results, bert_scores, meteor_scores, rouge_scores, exercise_matches)


def create_summary_sheet(wb, results, bert_scores, meteor_scores, rouge_scores, 
                         exercise_matches, calculator):
    """Create summary sheet with statistics."""
    summary_ws = wb.create_sheet("Summary")
    
    border = Border(
        left=Side(style='thin'),
        right=Side(style='thin'),
        top=Side(style='thin'),
        bottom=Side(style='thin')
    )
    
    summary_data = [
        ["Metric", "Value"],
        ["Total Samples", len(results)],
        ["Successful", sum(1 for r in results if r.get('status') == 'success')],
        ["Failed", sum(1 for r in results if r.get('status') == 'error')],
        ["", ""],
    ]
    
    # Add metric statistics
    if calculator.use_bert and bert_scores:
        summary_data.extend(create_metric_summary("BERT Similarity", bert_scores, 
                                                  BERT_GREEN_THRESHOLD, BERT_YELLOW_THRESHOLD))
    
    if meteor_scores:
        summary_data.extend(create_metric_summary("METEOR Score", meteor_scores,
                                                  METEOR_GREEN_THRESHOLD, METEOR_YELLOW_THRESHOLD))
    
    if rouge_scores:
        summary_data.extend(create_metric_summary("ROUGE-L Score", rouge_scores,
                                                  ROUGE_GREEN_THRESHOLD, ROUGE_YELLOW_THRESHOLD))
    
    # Exercise identification
    exercise_correct = sum(1 for match in exercise_matches if match)
    exercise_incorrect = len(exercise_matches) - exercise_correct
    exercise_accuracy = (exercise_correct / len(exercise_matches) * 100) if exercise_matches else 0
    
    summary_data.extend([
        ["Exercise Identification", ""],
        ["Correct", exercise_correct],
        ["Incorrect", exercise_incorrect],
        ["Accuracy (%)", round(exercise_accuracy, 2)],
        ["", ""],
    ])
    
    # Write summary data
    for row_idx, row_data in enumerate(summary_data, start=1):
        for col_idx, value in enumerate(row_data, start=1):
            cell = summary_ws.cell(row=row_idx, column=col_idx)
            cell.value = value
            cell.border = border
            
            if row_idx == 1 or (isinstance(value, str) and value and row_idx > 1 and col_idx == 1):
                cell.font = Font(bold=True)
    
    summary_ws.column_dimensions['A'].width = 30
    summary_ws.column_dimensions['B'].width = 15


def create_metric_summary(metric_name: str, scores: List[float], 
                         green_thresh: float, yellow_thresh: float) -> List[List]:
    """Create summary statistics for a metric."""
    green_count = sum(1 for s in scores if s >= green_thresh)
    yellow_count = sum(1 for s in scores if yellow_thresh <= s < green_thresh)
    red_count = sum(1 for s in scores if s < yellow_thresh)
    
    return [
        [metric_name, ""],
        ["Mean", round(np.mean(scores), 4)],
        ["Median", round(np.median(scores), 4)],
        ["Std Dev", round(np.std(scores), 4)],
        ["Min", round(np.min(scores), 4)],
        ["Max", round(np.max(scores), 4)],
        [f"Green (≥{green_thresh})", green_count],
        [f"Yellow ({yellow_thresh}-{green_thresh})", yellow_count],
        [f"Red (<{yellow_thresh})", red_count],
        ["", ""],
    ]


def print_summary(results, bert_scores, meteor_scores, rouge_scores, exercise_matches):
    """Print summary statistics to console."""
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('status') == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.get('status') == 'error')}")
    
    if bert_scores:
        print(f"\nBERT Similarity:")
        print(f"  Mean: {np.mean(bert_scores):.4f}")
        print(f"  Median: {np.median(bert_scores):.4f}")
        print(f"  Std Dev: {np.std(bert_scores):.4f}")
    
    if meteor_scores:
        print(f"\nMETEOR Score:")
        print(f"  Mean: {np.mean(meteor_scores):.4f}")
        print(f"  Median: {np.median(meteor_scores):.4f}")
        print(f"  Std Dev: {np.std(meteor_scores):.4f}")
    
    if rouge_scores:
        print(f"\nROUGE-L Score:")
        print(f"  Mean: {np.mean(rouge_scores):.4f}")
        print(f"  Median: {np.median(rouge_scores):.4f}")
        print(f"  Std Dev: {np.std(rouge_scores):.4f}")
    
    if exercise_matches:
        exercise_correct = sum(1 for match in exercise_matches if match)
        exercise_accuracy = (exercise_correct / len(exercise_matches) * 100)
        print(f"\nExercise Identification:")
        print(f"  Accuracy: {exercise_accuracy:.2f}%")
    
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate test evaluation report for NVILA/VILA models"
    )
    parser.add_argument("--predictions", type=str, required=True,
                       help="Path to predictions CSV file (from inference.py)")
    parser.add_argument("--base-predictions", type=str, default=None,
                       help="Path to base model predictions CSV file (optional)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output Excel file path (default: auto-generated)")
    parser.add_argument("--no-bert", action="store_true",
                       help="Skip BERT similarity computation")
    
    args = parser.parse_args()
    
    # Set default output path
    pred_path = Path(args.predictions)
    if args.output is None:
        args.output = str(pred_path.parent / f"{pred_path.stem}_report.xlsx")
    
    print(f"Excel output will be saved to: {args.output}")
    
    # Load predictions
    print(f"\nLoading fine-tuned model predictions from: {args.predictions}")
    results = load_csv_predictions(args.predictions)
    print(f"✓ Loaded {len(results)} predictions")
    
    # Load base predictions if provided
    base_predictions = None
    if args.base_predictions:
        print(f"\nLoading base model predictions from: {args.base_predictions}")
        base_predictions = load_csv_predictions(args.base_predictions)
        print(f"✓ Loaded {len(base_predictions)} base model predictions")
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(use_bert=not args.no_bert)
    
    # Generate report
    create_excel_report(results, args.output, calculator, base_predictions)
    
    print("\n✓ Report generated successfully!")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Generate Test Evaluation Report for NVILA/VILA Models

This script processes test inference results and generates comprehensive Excel reports
with multiple similarity metrics including BERT, METEOR, ROUGE-L, and optional LLM judging.

Usage:
    python generate_vila_report.py --predictions test_predictions.json
    python generate_vila_report.py --predictions test_predictions.json --output report.xlsx --csv-output report.csv
    python generate_vila_report.py --predictions test_predictions.json --no-bert --no-llm-judge
"""

import json
import argparse
import numpy as np
import csv
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import openpyxl
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, Reference
from openpyxl.chart.legend import Legend
from sklearn.metrics.pairwise import cosine_similarity
import evaluate
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Metric thresholds
ROUGE_GREEN_THRESHOLD = 0.5
ROUGE_YELLOW_THRESHOLD = 0.2

BERT_GREEN_THRESHOLD = 0.7
BERT_YELLOW_THRESHOLD = 0.4

METEOR_GREEN_THRESHOLD = 0.5
METEOR_YELLOW_THRESHOLD = 0.2

LLM_GREEN_THRESHOLD = 4.0
LLM_YELLOW_THRESHOLD = 3.0


class MetricsCalculator:
    """Handles all metric calculations."""
    
    def __init__(self, use_bert: bool = True, use_llm_judge: bool = False):
        self.use_bert = use_bert
        self.use_llm_judge = use_llm_judge
        
        # Load models
        self.bert_model = self._load_bert_model() if use_bert else None
        self.meteor_metric = self._load_metric('meteor')
        self.rouge_metric = self._load_metric('rouge')
        self.llm_tokenizer, self.llm_model = self._load_llm_judge() if use_llm_judge else (None, None)
    
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
    
    def _load_llm_judge(self) -> Tuple[Optional[AutoTokenizer], Optional[AutoModelForCausalLM]]:
        """Load Mixtral-Instruct for LLM judging."""
        print("\nLoading LLM Judge (Mixtral-8x7B-Instruct-v0.1)...")
        print("This may take a few minutes on first run...")
        
        try:
            model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                load_in_8bit=True
            )
            
            print("✓ LLM Judge loaded successfully")
            return tokenizer, model
            
        except Exception as e:
            print(f"⚠ Warning: Could not load LLM judge: {e}")
            print("  LLM Accuracy scores will be skipped")
            return None, None
    
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
    
    def compute_llm_accuracy(self, ground_truth: str, prediction: str) -> float:
        """Use Mixtral as judge to score prediction (1-5 scale)."""
        if not ground_truth or not prediction:
            return 0.0
        
        if self.llm_tokenizer is None or self.llm_model is None:
            return 0.0
        
        try:
            prompt = f"""[INST] You are an expert evaluator for exercise feedback quality.

Given a ground-truth feedback and a predicted feedback for a physiotherapy exercise video, rate the predicted feedback on a scale of 1-5 for holistic accuracy and usefulness.

Rating Scale:
5 - Excellent: Predicted feedback is highly accurate, covers all key points, and is very useful
4 - Good: Predicted feedback is mostly accurate with minor omissions, still quite useful
3 - Moderate: Predicted feedback has some accuracy but misses important details
2 - Poor: Predicted feedback has major inaccuracies or missing critical information
1 - Very Poor: Predicted feedback is largely incorrect or not useful

Ground-truth feedback:
{ground_truth}

Predicted feedback:
{prediction}

Provide only a single number (1-5) as your rating. [/INST]

Rating:"""
            
            inputs = self.llm_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            inputs = {k: v.to(self.llm_model.device) for k, v in inputs.items()}
            
            with torch.inference_mode():
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=10,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=self.llm_tokenizer.eos_token_id
                )
            
            response = self.llm_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            
            match = re.search(r'\b([1-5])\b', response)
            if match:
                return float(match.group(1))
            else:
                print(f"⚠ Could not parse LLM response: {response[:100]}")
                return 3.0
                
        except Exception as e:
            print(f"⚠ Error computing LLM accuracy: {e}")
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


def create_csv_report(results: List[Dict], output_path: str, calculator: MetricsCalculator):
    """Create a CSV report with similarity scores."""
    print(f"\nGenerating CSV report...")
    
    with open(output_path, mode='w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        
        # Write headers
        headers = ["ID", "Video Path", "Ground Truth", "Prediction"]
        
        if calculator.use_bert:
            headers.append("BERT Similarity")
        
        headers.extend(["METEOR Score", "ROUGE-L Score"])
        
        if calculator.use_llm_judge:
            headers.append("LLM Accuracy (1-5)")
        
        headers.append("Exercise Match")
        
        writer.writerow(headers)
        
        for idx, result in enumerate(results, start=1):
            video_path = result.get('video_path', '')
            ground_truth = result.get('ground_truth', '')
            prediction = result.get('prediction', '')
            
            row = [idx, video_path, ground_truth, prediction]
            
            if calculator.use_bert:
                bert_sim = calculator.compute_bert_similarity(ground_truth, prediction)
                row.append(round(bert_sim, 4))
            
            meteor_score = calculator.compute_meteor(ground_truth, prediction)
            rouge_score = calculator.compute_rouge(ground_truth, prediction)
            row.extend([round(meteor_score, 4), round(rouge_score, 4)])
            
            if calculator.use_llm_judge:
                llm_score = calculator.compute_llm_accuracy(ground_truth, prediction)
                row.append(round(llm_score, 2))
            
            exercise_match = check_exercise_match(ground_truth, prediction)
            row.append("TRUE" if exercise_match else "FALSE")
            
            writer.writerow(row)
    
    print(f"✓ CSV report saved to: {output_path}")


def apply_color_coding(cell, score: float, metric_type: str):
    """Apply color coding based on score and metric type."""
    thresholds = {
        'bert': (BERT_GREEN_THRESHOLD, BERT_YELLOW_THRESHOLD),
        'meteor': (METEOR_GREEN_THRESHOLD, METEOR_YELLOW_THRESHOLD),
        'rouge': (ROUGE_GREEN_THRESHOLD, ROUGE_YELLOW_THRESHOLD),
        'llm': (LLM_GREEN_THRESHOLD, LLM_YELLOW_THRESHOLD)
    }
    
    green_thresh, yellow_thresh = thresholds.get(metric_type, (0.5, 0.2))
    
    if score >= green_thresh:
        cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
    elif score >= yellow_thresh:
        cell.fill = PatternFill(start_color="FFEB9C", end_color="FFEB9C", fill_type="solid")
    else:
        cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")


def create_excel_report(results: List[Dict], output_path: str, calculator: MetricsCalculator):
    """Create comprehensive Excel report with all metrics and visualizations."""
    print(f"\nGenerating Excel report...")
    
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Evaluation Results"
    
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
    headers = ["ID", "Video Path", "Ground Truth", "Prediction"]
    
    if calculator.use_bert:
        headers.append("BERT Similarity")
    
    headers.extend(["METEOR Score", "ROUGE-L Score"])
    
    if calculator.use_llm_judge:
        headers.append("LLM Accuracy (1-5)")
    
    headers.append("Exercise Match")
    
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
    if calculator.use_bert:
        ws.column_dimensions[get_column_letter(col_idx)].width = 15
        col_idx += 1
    
    ws.column_dimensions[get_column_letter(col_idx)].width = 15  # METEOR
    ws.column_dimensions[get_column_letter(col_idx + 1)].width = 15  # ROUGE
    col_idx += 2
    
    if calculator.use_llm_judge:
        ws.column_dimensions[get_column_letter(col_idx)].width = 18  # LLM
        col_idx += 1
    
    ws.column_dimensions[get_column_letter(col_idx)].width = 15  # Exercise Match
    
    ws.freeze_panes = "A2"
    
    # Process results
    print(f"Processing {len(results)} results...")
    
    bert_scores = []
    meteor_scores = []
    rouge_scores = []
    llm_scores = []
    exercise_matches = []
    
    for idx, result in enumerate(results, start=1):
        row = idx + 1
        
        video_path = result.get('video_path', '')
        ground_truth = result.get('ground_truth', '')
        prediction = result.get('prediction', '')
        
        # Compute metrics
        bert_sim = None
        if calculator.use_bert:
            bert_sim = calculator.compute_bert_similarity(ground_truth, prediction)
            bert_scores.append(bert_sim)
        
        meteor_sim = calculator.compute_meteor(ground_truth, prediction)
        meteor_scores.append(meteor_sim)
        
        rouge_sim = calculator.compute_rouge(ground_truth, prediction)
        rouge_scores.append(rouge_sim)
        
        llm_score = 0.0
        if calculator.use_llm_judge:
            llm_score = calculator.compute_llm_accuracy(ground_truth, prediction)
            if llm_score > 0:
                llm_scores.append(llm_score)
        
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
        
        llm_col_idx = None
        if calculator.use_llm_judge:
            llm_col_idx = col
            if llm_score > 0:
                ws.cell(row=row, column=col).value = round(llm_score, 2)
            col += 1
        
        exercise_col_idx = col
        ws.cell(row=row, column=col).value = "TRUE" if exercise_match else "FALSE"
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
            elif llm_col_idx and c == llm_col_idx and llm_score > 0:
                apply_color_coding(cell, llm_score, 'llm')
            elif c == exercise_col_idx:
                if exercise_match:
                    cell.fill = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
                else:
                    cell.fill = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
    
    # Create summary sheet
    create_summary_sheet(wb, results, bert_scores, meteor_scores, rouge_scores, 
                        llm_scores, exercise_matches, calculator)
    
    # Save workbook
    wb.save(output_path)
    print(f"✓ Excel report saved to: {output_path}")
    
    # Print summary
    print_summary(results, bert_scores, meteor_scores, rouge_scores, llm_scores, exercise_matches)


def create_summary_sheet(wb, results, bert_scores, meteor_scores, rouge_scores, 
                         llm_scores, exercise_matches, calculator):
    """Create summary sheet with statistics and charts."""
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
    chart_positions = {}
    
    if calculator.use_bert and bert_scores:
        chart_positions['bert'] = len(summary_data) + 6
        summary_data.extend(create_metric_summary("BERT Similarity", bert_scores, 
                                                  BERT_GREEN_THRESHOLD, BERT_YELLOW_THRESHOLD))
    
    if meteor_scores:
        chart_positions['meteor'] = len(summary_data) + 6
        summary_data.extend(create_metric_summary("METEOR Score", meteor_scores,
                                                  METEOR_GREEN_THRESHOLD, METEOR_YELLOW_THRESHOLD))
    
    if rouge_scores:
        chart_positions['rouge'] = len(summary_data) + 6
        summary_data.extend(create_metric_summary("ROUGE-L Score", rouge_scores,
                                                  ROUGE_GREEN_THRESHOLD, ROUGE_YELLOW_THRESHOLD))
    
    if calculator.use_llm_judge and llm_scores:
        chart_positions['llm'] = len(summary_data) + 6
        summary_data.extend(create_metric_summary("LLM Accuracy (1-5)", llm_scores,
                                                  LLM_GREEN_THRESHOLD, LLM_YELLOW_THRESHOLD))
    
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
    
    # Add charts
    chart_row = 2
    for metric_name, start_row in chart_positions.items():
        chart = create_distribution_chart(summary_ws, metric_name, start_row)
        summary_ws.add_chart(chart, f"D{chart_row}")
        chart_row += 15


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


def create_distribution_chart(ws, metric_name: str, start_row: int) -> BarChart:
    """Create a bar chart for metric distribution."""
    from openpyxl.chart.series import DataPoint
    from openpyxl.chart.shapes import GraphicalProperties
    
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = f"{metric_name} Distribution"
    chart.y_axis.title = "Count"
    chart.x_axis.title = "Category"
    
    chart.x_axis.delete = False
    chart.y_axis.delete = False
    chart.x_axis.majorTickMark = "out"
    chart.y_axis.majorTickMark = "out"
    
    chart.legend = Legend()
    chart.legend.position = "b"
    
    data = Reference(ws, min_col=2, min_row=start_row, max_row=start_row + 2)
    cats = Reference(ws, min_col=1, min_row=start_row, max_row=start_row + 2)
    
    chart.add_data(data, titles_from_data=False)
    chart.set_categories(cats)
    chart.shape = 4
    chart.width = 12
    chart.height = 8
    
    # Color the bars
    series = chart.series[0]
    
    pt_green = DataPoint(idx=0)
    pt_green.graphicalProperties = GraphicalProperties()
    pt_green.graphicalProperties.solidFill = "00B050"
    series.data_points.append(pt_green)
    
    pt_yellow = DataPoint(idx=1)
    pt_yellow.graphicalProperties = GraphicalProperties()
    pt_yellow.graphicalProperties.solidFill = "FFC000"
    series.data_points.append(pt_yellow)
    
    pt_red = DataPoint(idx=2)
    pt_red.graphicalProperties = GraphicalProperties()
    pt_red.graphicalProperties.solidFill = "FF0000"
    series.data_points.append(pt_red)
    
    return chart


def print_summary(results, bert_scores, meteor_scores, rouge_scores, llm_scores, exercise_matches):
    """Print summary statistics to console."""
    print(f"\n{'='*60}")
    print("Evaluation Summary")
    print(f"{'='*60}")
    print(f"Total samples: {len(results)}")
    print(f"Successful: {sum(1 for r in results if r.get('status', 'success') == 'success')}")
    print(f"Failed: {sum(1 for r in results if r.get('status', 'success') == 'error')}")
    
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
    
    if llm_scores:
        print(f"\nLLM Accuracy (1-5 scale):")
        print(f"  Mean: {np.mean(llm_scores):.2f}")
        print(f"  Median: {np.median(llm_scores):.2f}")
        print(f"  Std Dev: {np.std(llm_scores):.2f}")
    
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
                       help="Path to predictions JSON file")
    parser.add_argument("--output", type=str, default=None,
                       help="Output Excel file path (default: auto-generated)")
    parser.add_argument("--csv-output", type=str, default=None,
                       help="Output CSV file path (default: auto-generated)")
    parser.add_argument("--no-bert", action="store_true",
                       help="Skip BERT similarity computation")
    parser.add_argument("--no-llm-judge", action="store_true",
                       help="Skip LLM judge evaluation")
    
    args = parser.parse_args()
    
    # Set default output paths
    pred_path = Path(args.predictions)
    if args.output is None:
        args.output = str(pred_path.parent / f"{pred_path.stem}_report.xlsx")
    if args.csv_output is None:
        args.csv_output = str(pred_path.parent / f"{pred_path.stem}_report.csv")
    
    print(f"Excel output: {args.output}")
    print(f"CSV output: {args.csv_output}")
    
    # Load predictions
    print(f"\nLoading predictions from: {args.predictions}")
    with open(args.predictions, 'r') as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} predictions")
    
    # Initialize metrics calculator
    calculator = MetricsCalculator(
        use_bert=not args.no_bert,
        use_llm_judge=not args.no_llm_judge
    )
    
    # Generate reports
    create_excel_report(results, args.output, calculator)
    create_csv_report(results, args.csv_output, calculator)
    
    print("\n✓ All reports generated successfully!")


if __name__ == "__main__":
    main()
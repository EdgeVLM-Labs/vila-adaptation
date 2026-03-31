"""
Exercise Identification Evaluation Pipeline

This script evaluates whether a fine-tuned model correctly identifies exercises
from video inputs by comparing model outputs against ground truth using semantic similarity.

Metric Used: Sentence-BERT (SBERT) Cosine Similarity
- Model: all-MiniLM-L6-v2 (fast and accurate for short text comparison)
- Threshold: 0.75 (exercises with similarity >= 0.75 are considered correctly identified)

Threshold Justification:
- 0.75-1.0: Strong semantic match (e.g., "heel lift" vs "calf raise")
- 0.50-0.75: Moderate similarity (possibly related exercises)
- <0.50: Different exercises

Author: Exercise Model Evaluation Team
Date: January 2026
"""

import pandas as pd
import re
from typing import List, Tuple, Dict
from sentence_transformers import SentenceTransformer, util
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ExerciseEvaluationPipeline:
    """Pipeline for evaluating exercise identification in model outputs"""
    
    # Configuration constants
    SBERT_MODEL = 'all-MiniLM-L6-v2'
    SIMILARITY_THRESHOLD = 0.75
    MIN_NGRAM = 1
    MAX_NGRAM = 8
    
    def __init__(self, csv_path: str):
        """
        Initialize the evaluation pipeline
        
        Args:
            csv_path: Path to input CSV file
        """
        self.csv_path = csv_path
        self.model = SentenceTransformer(self.SBERT_MODEL)
        logger.info(f"Loaded SBERT model: {self.SBERT_MODEL}")
        
    def load_data(self) -> pd.DataFrame:
        """Load CSV and create DataFrame"""
        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        logger.info(f"Loaded {len(df)} rows")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows based on video column"""
        initial_count = len(df)
        df = df.drop_duplicates(subset=['video'], keep='first')
        removed_count = initial_count - len(df)
        logger.info(f"Removed {removed_count} duplicate videos. Remaining: {len(df)}")
        return df
    
    def extract_exercise_from_ground_truth(self, ground_truth: str) -> str:
        """
        Extract exercise name from ground truth format: "exercise - feedback"
        
        Args:
            ground_truth: Ground truth string in format "exercise - feedback"
            
        Returns:
            Extracted exercise name
        """
        if pd.isna(ground_truth):
            return ""
        
        ground_truth = str(ground_truth).strip()
        
        # Split by " - " separator
        if " - " in ground_truth:
            exercise = ground_truth.split(" - ")[0].strip()
            return exercise
        
        # If no separator, return the whole string
        return ground_truth
    
    def clean_model_output(self, model_output: str) -> str:
        """
        Clean model output by removing logging information
        
        Args:
            model_output: Raw model output with potential logging
            
        Returns:
            Cleaned model output
        """
        if pd.isna(model_output):
            return ""
        
        model_output = str(model_output).strip()
        
        # Remove lines that start with date/timestamp and logging info
        # Pattern: [YYYY-MM-DD HH:MM:SS,mmm] [LEVEL] [file.py:line:function] message
        pattern = r'\[\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d{3}\]\s+\[.*?\]\s+\[.*?\].*?(?=\n|$)'
        cleaned = re.sub(pattern, '', model_output, flags=re.MULTILINE)
        
        # Remove any remaining empty lines
        cleaned = re.sub(r'\n\s*\n', '\n', cleaned)
        
        return cleaned.strip()
    
    def generate_ngrams(self, text: str, min_n: int, max_n: int) -> List[str]:
        """
        Generate n-grams from text
        
        Args:
            text: Input text
            min_n: Minimum n-gram size
            max_n: Maximum n-gram size
            
        Returns:
            List of n-gram phrases
        """
        if not text:
            return []
        
        # Split into sentences
        sentences = re.split(r'[.!?]+', text)
        ngrams = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Split by common separators
            parts = re.split(r'\s+-\s+|,\s+', sentence)
            
            for part in parts:
                words = part.strip().split()
                
                # Generate n-grams
                for n in range(min_n, min(len(words) + 1, max_n + 1)):
                    for i in range(len(words) - n + 1):
                        ngram = ' '.join(words[i:i+n])
                        # Basic filtering
                        if len(ngram) > 2:  # Avoid very short phrases
                            ngrams.append(ngram.strip())
        
        return list(set(ngrams))  # Remove duplicates
    
    def calculate_best_similarity(self, gt_exercise: str, model_output: str) -> Tuple[float, str, int]:
        """
        Calculate best similarity score across all n-grams
        
        Args:
            gt_exercise: Ground truth exercise name
            model_output: Cleaned model output
            
        Returns:
            Tuple of (best_score, best_matching_ngram, ngram_size)
        """
        if not gt_exercise or not model_output:
            return 0.0, "", 0
        
        # Generate all n-grams
        ngrams = self.generate_ngrams(model_output, self.MIN_NGRAM, self.MAX_NGRAM)
        
        if not ngrams:
            return 0.0, "", 0
        
        # Encode ground truth once
        gt_embedding = self.model.encode(gt_exercise, convert_to_tensor=True)
        
        # Encode all n-grams in batch (efficient)
        ngram_embeddings = self.model.encode(ngrams, convert_to_tensor=True)
        
        # Calculate cosine similarities
        similarities = util.cos_sim(gt_embedding, ngram_embeddings)[0]
        
        # Find best match
        best_idx = similarities.argmax().item()
        best_score = similarities[best_idx].item()
        best_ngram = ngrams[best_idx]
        ngram_size = len(best_ngram.split())
        
        return best_score, best_ngram, ngram_size
    
    def evaluate_row(self, row: pd.Series) -> Dict:
        """
        Evaluate a single row
        
        Args:
            row: DataFrame row
            
        Returns:
            Dictionary with evaluation results
        """
        # Extract exercise from ground truth
        gt_exercise = self.extract_exercise_from_ground_truth(row['ground_truth'])
        
        # Clean model output
        cleaned_output = self.clean_model_output(row['model_output'])
        
        # Calculate best similarity
        best_score, best_ngram, ngram_size = self.calculate_best_similarity(
            gt_exercise, cleaned_output
        )
        
        # Determine if correctly identified
        is_correct = best_score >= self.SIMILARITY_THRESHOLD
        
        return {
            'gt_exercise': gt_exercise,
            'cleaned_model_output': cleaned_output,
            'best_matching_ngram': best_ngram,
            'similarity_score': round(best_score, 4),
            'ngram_size': ngram_size,
            'exercise_identified_correctly': is_correct
        }
    
    def process_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process entire DataFrame
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with evaluation results
        """
        logger.info("Processing DataFrame...")
        
        results = []
        for idx, row in df.iterrows():
            result = self.evaluate_row(row)
            results.append(result)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(df)} rows")
        
        # Add results as new columns
        df['gt_exercise'] = [r['gt_exercise'] for r in results]
        df['cleaned_model_output'] = [r['cleaned_model_output'] for r in results]
        df['best_matching_ngram'] = [r['best_matching_ngram'] for r in results]
        df['similarity_score'] = [r['similarity_score'] for r in results]
        df['ngram_size'] = [r['ngram_size'] for r in results]
        df['exercise_identified_correctly'] = [r['exercise_identified_correctly'] for r in results]
        
        logger.info("Processing complete")
        return df
    
    def create_per_exercise_breakdown(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create per-exercise breakdown showing correct/total counts
        
        Args:
            df: DataFrame with evaluation results
            
        Returns:
            DataFrame with per-exercise breakdown
        """
        logger.info("Creating per-exercise breakdown...")
        
        # Group by ground truth exercise
        breakdown = df.groupby('gt_exercise').agg(
            total_count=('exercise_identified_correctly', 'count'),
            correct_count=('exercise_identified_correctly', 'sum')
        ).reset_index()
        
        # Rename column for clarity
        breakdown.rename(columns={'gt_exercise': 'Per-Exercise Breakdown'}, inplace=True)
        
        # Create the format "correct/total" similar to the image
        breakdown['Count'] = breakdown.apply(
            lambda row: f"{int(row['correct_count'])}/{int(row['total_count'])}", 
            axis=1
        )
        
        # Calculate accuracy percentage for each exercise
        breakdown['Accuracy (%)'] = (breakdown['correct_count'] / breakdown['total_count'] * 100).round(2)
        
        # Select and reorder columns
        breakdown = breakdown[['Per-Exercise Breakdown', 'Count', 'Accuracy (%)']]
        
        # Sort by exercise name
        breakdown = breakdown.sort_values('Per-Exercise Breakdown').reset_index(drop=True)
        
        logger.info(f"Created breakdown for {len(breakdown)} exercises")
        
        return breakdown
    
    def save_results(self, df: pd.DataFrame, breakdown_df: pd.DataFrame, output_path: str) -> None:
        """
        Save results to Excel file with multiple sheets or separate CSV files
        
        Args:
            df: DataFrame with detailed results
            breakdown_df: DataFrame with per-exercise breakdown
            output_path: Output file path
        """
        output_path = Path(output_path)
        
        # Try to save as Excel file with multiple sheets
        try:
            excel_path = output_path.with_suffix('.xlsx')
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                df.to_csv(output_path, index=False)
                df.to_excel(writer, sheet_name='Detailed Results', index=False)
                breakdown_df.to_excel(writer, sheet_name='Per-Exercise Breakdown', index=False)
            
            logger.info(f"Results saved to Excel file: {excel_path}")
            logger.info(f"  - Sheet 1: Detailed Results")
            logger.info(f"  - Sheet 2: Per-Exercise Breakdown")
            
        except ImportError:
            # If openpyxl is not available, save as separate CSV files
            logger.warning("openpyxl not available. Saving as separate CSV files.")
            
            # Save detailed results
            df.to_csv(output_path, index=False)
            logger.info(f"Detailed results saved to: {output_path}")
            
            # Save breakdown
            breakdown_path = output_path.parent / f"{output_path.stem}_breakdown.csv"
            breakdown_df.to_csv(breakdown_path, index=False)
            logger.info(f"Per-exercise breakdown saved to: {breakdown_path}")
    
    def print_summary(self, df: pd.DataFrame) -> None:
        """
        Print evaluation summary
        
        Args:
            df: DataFrame with results
        """
        total = len(df)
        correct = df['exercise_identified_correctly'].sum()
        accuracy = (correct / total * 100) if total > 0 else 0
        
        avg_score = df['similarity_score'].mean()
        avg_score_correct = df[df['exercise_identified_correctly']]['similarity_score'].mean()
        avg_score_incorrect = df[~df['exercise_identified_correctly']]['similarity_score'].mean()
        
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Total samples: {total}")
        print(f"Correctly identified: {correct}")
        print(f"Incorrectly identified: {total - correct}")
        print(f"Accuracy: {accuracy:.2f}%")
        print(f"\nSimilarity Scores:")
        print(f"  Overall average: {avg_score:.4f}")
        print(f"  When correct: {avg_score_correct:.4f}")
        print(f"  When incorrect: {avg_score_incorrect:.4f}")
        print(f"\nThreshold used: {self.SIMILARITY_THRESHOLD}")
        print(f"Model used: {self.SBERT_MODEL}")
        print("=" * 70 + "\n")
    
    def print_breakdown_summary(self, breakdown_df: pd.DataFrame) -> None:
        """
        Print per-exercise breakdown summary
        
        Args:
            breakdown_df: DataFrame with per-exercise breakdown
        """
        print("\n" + "=" * 70)
        print("PER-EXERCISE BREAKDOWN")
        print("=" * 70)
        print(breakdown_df.to_string(index=False))
        print("=" * 70 + "\n")
    
    def run(self, output_path: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Run the complete pipeline
        
        Args:
            output_path: Path for output file (if None, auto-generates)
            
        Returns:
            Tuple of (detailed_results_df, breakdown_df)
        """
        # Step 1: Load data
        df = self.load_data()
        
        # Step 2: Remove duplicates
        df = self.remove_duplicates(df)
        
        # Step 3-6: Process all rows
        df = self.process_dataframe(df)
        
        # Step 7: Create per-exercise breakdown
        breakdown_df = self.create_per_exercise_breakdown(df)
        
        # Step 8: Save results
        if output_path is None:
            input_path = Path(self.csv_path)
            output_path = input_path.parent / f"{input_path.stem}_evaluated.csv"
        
        self.save_results(df, breakdown_df, output_path)
        
        # Print summaries
        self.print_summary(df)
        self.print_breakdown_summary(breakdown_df)
        
        return df, breakdown_df


def main():
    """Main execution function"""
    # Configuration
    INPUT_CSV = "2_finetuned_inference_results.csv"  # Change this to your CSV file path
    OUTPUT_CSV = "n_grams_exercise_evaluation_results.csv"  # Change this to desired output path
    
    # Run pipeline
    pipeline = ExerciseEvaluationPipeline(INPUT_CSV)
    results_df, breakdown_df = pipeline.run(OUTPUT_CSV)
    
    # Optional: Show sample results
    print("\nSample Detailed Results (first 5 rows):")
    print(results_df[['video', 'gt_exercise', 'best_matching_ngram', 
                      'similarity_score', 'exercise_identified_correctly']].head())
    
    print("\nSample Breakdown (first 5 exercises):")
    print(breakdown_df.head())


if __name__ == "__main__":
    main()
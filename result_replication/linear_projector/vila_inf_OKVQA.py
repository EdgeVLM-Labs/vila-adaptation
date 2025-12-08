import random
import re
import subprocess
import tempfile
import os
from typing import List

from datasets import load_dataset
from openpyxl import Workbook
from PIL import Image

# ======================================
# CONFIG
# ======================================
MODEL_PATH = "Efficient-Large-Model/VILA1.5-7b"
CONV_MODE = "vicuna_v1"
NUM_QUESTIONS = 100               # <---- total number of questions to evaluate
SPLIT = "train"                   # or "validation" if you prefer
# OUTPUT_XLSX will be set after INFERENCE_MODE is defined
SEED = 42

# ======================================
# INFERENCE MODE CONFIGURATION
# ======================================
INFERENCE_MODE = "0-shot"  # Options: "0-shot", "4-shot"
NUM_SHOTS = 4  # Number of examples for few-shot (only used when INFERENCE_MODE is \"4-shot\")\n\n# Set output filename based on inference mode\nOUTPUT_XLSX = f\"vila_OKVQA_{INFERENCE_MODE}_evaluation.xlsx\"

random.seed(SEED)

# Set output filename based on inference mode
OUTPUT_XLSX = f"vila_OKVQA_{INFERENCE_MODE}_evaluation.xlsx"

# ======================================
# 1. LOAD DATASET (A-OKVQA)
# ======================================
print("Loading HuggingFaceM4/A-OKVQA...")

dataset_name = "HuggingFaceM4/A-OKVQA"
dataset = load_dataset(dataset_name)

print("Available splits:", dataset.keys())
data_split = dataset[SPLIT]
total_questions = len(data_split)
print(f"Found {total_questions} questions in split '{SPLIT}'.")

# Select random subset of questions
if NUM_QUESTIONS > total_questions:
    NUM_QUESTIONS = total_questions
selected_indices = random.sample(range(total_questions), NUM_QUESTIONS)
print(f"\nSelected {NUM_QUESTIONS} random questions for evaluation.")

# ======================================
# FEW-SHOT EXAMPLE SELECTION
# ======================================
def select_few_shot_examples(dataset, num_shots: int, exclude_indices: List[int]) -> List[dict]:
    """
    Select few-shot examples from the dataset, excluding the test indices.
    """
    available_indices = [i for i in range(len(dataset)) if i not in exclude_indices]
    if len(available_indices) < num_shots:
        print(f"Warning: Only {len(available_indices)} examples available for few-shot, using all.")
        num_shots = len(available_indices)
    
    shot_indices = random.sample(available_indices, num_shots)
    examples = []
    
    for idx in shot_indices:
        row = dataset[idx]
        # Get the correct answer from choices using correct_choice_idx
        correct_answer = row["choices"][row["correct_choice_idx"]]
        examples.append({
            "question": row["question"],
            "choices": row["choices"],
            "answer": correct_answer,
            "image": row["image"]  # Include image for few-shot examples
        })
    
    return examples

# Select few-shot examples if needed
few_shot_examples = []
if INFERENCE_MODE == "4-shot":
    few_shot_examples = select_few_shot_examples(data_split, NUM_SHOTS, selected_indices)
    print(f"Selected {len(few_shot_examples)} few-shot examples.")

# ======================================
# 2. PROMPT BUILDER
# ======================================
def build_prompt(question: str, choices: List[str], few_shot_examples: List[dict] = None) -> str:
    """
    Build a prompt for A-OKVQA that supports both 0-shot and few-shot inference.
    Note: For few-shot, images need to be handled separately in the inference pipeline.
    """
    prompt = ""
    
    # Add few-shot examples if provided (text-only, images handled separately)
    if few_shot_examples:
        prompt += "Here are some examples of visual question answering:\n\n"
        for i, example in enumerate(few_shot_examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Question: {example['question']}\n"
            prompt += f"Possible answers: {', '.join(example['choices'])}\n"
            prompt += f"Answer: {example['answer']}\n\n"
        
        prompt += "Now, looking at the current image, answer the following question:\n\n"
    else:
        prompt += "Looking at the image, "
    
    # Add the main question
    prompt += f"Question: {question}\n\nPossible answers: {', '.join(choices)}\n\n"
    # prompt += (
    #     "Please answer the question naturally based on what you see in the image. "
    #     "You can use one of the possible answers or provide your own answer.\nAnswer:"
    # )

    print("*******[prompt]:", prompt)
    return prompt

# ======================================
# 3. CALL VILA VIA CLI (WITH IMAGE)
# ======================================
def run_vila_infer(prompt: str, image) -> str:
    """
    Call the VILA CLI with both text and image.
    Saves image temporarily and passes it to vila-infer.
    """
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
        temp_image_path = temp_file.name
        # Convert PIL Image to RGB if necessary and save
        if hasattr(image, 'convert'):
            image.convert('RGB').save(temp_image_path, 'PNG')
        else:
            # If it's already a file path, copy it
            image.save(temp_image_path)
    
    try:
        cmd = [
            "vila-infer",
            "--model-path", MODEL_PATH,
            "--conv-mode", CONV_MODE,
            "--text", prompt,
            "--media", temp_image_path,
        ]

        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        if result.returncode != 0:
            print("Error running vila-infer:")
            print(result.stderr)
            return ""

        return result.stdout
    
    finally:
        # Clean up temporary file
        try:
            os.unlink(temp_image_path)
        except:
            pass

# ======================================
# 4. EXTRACT AND EVALUATE ANSWER
# ======================================
def extract_answer(output_text: str, choices: List[str]) -> str:
    """
    Extract the predicted answer from VILA's natural language output.
    No letter parsing - just clean text extraction.
    """
    if not output_text.strip():
        return ""

    # Clean output text - get the main response
    lines = [l.strip() for l in output_text.splitlines() if l.strip()]
    if not lines:
        return ""
    
    # Usually the answer is in the last meaningful line
    predicted_answer = lines[-1]
    
    # Remove common prefixes that models might add
    prefixes_to_remove = [
        "answer:", "the answer is:", "answer is:", "the answer is", 
        "answer is", "response:", "my answer is:", "i think:", 
        "i believe:", "the correct answer is:", "correct answer:"
    ]
    print("*******[predicted]: ", predicted_answer)
    
    predicted_lower = str(predicted_answer).lower().strip()
    for prefix in prefixes_to_remove:
        if predicted_lower.startswith(prefix):
            predicted_answer = predicted_answer[len(prefix):].strip()
            break
    
    # Remove punctuation at the end
    predicted_answer = predicted_answer.rstrip('.,!?;:')
    
    return predicted_answer.strip()

def check_answer_correct(predicted_answer: str, direct_answers: str, choices: List[str]) -> bool:
    """
    Check if the predicted answer matches any of the direct answers or choices.
    Uses flexible matching for natural language responses.
    """
    import ast
    
    if not predicted_answer.strip():
        return False
    
    predicted_lower = predicted_answer.lower().strip()
    
    # Debug output to understand matching
    print(f"    Checking: '{predicted_answer[:50]}...' vs direct_answers: {direct_answers}")
    print(f"    Choices: {choices}")
    
    # First, check against direct answers
    try:
        answer_list = ast.literal_eval(direct_answers)
        for answer in answer_list:
            answer_lower = answer.lower().strip()
            # Be more strict - only exact matches or if answer is clearly contained
            if (predicted_lower == answer_lower or 
                (len(answer_lower) > 2 and answer_lower in predicted_lower and len(answer_lower) >= len(predicted_lower) * 0.3)):
                print(f"    MATCH found with direct answer: '{answer}'")
                return True
    except:
        # Fallback for direct answers parsing
        if direct_answers and predicted_lower in direct_answers.lower():
            print(f"    MATCH found in direct_answers string")
            return True
    
    # Then check against provided choices - be more strict here too
    for choice in choices:
        choice_lower = choice.lower().strip()
        # Only match if there's significant overlap
        if (predicted_lower == choice_lower or 
            (len(choice_lower) > 2 and choice_lower in predicted_lower and len(choice_lower) >= len(predicted_lower) * 0.3)):
            print(f"    MATCH found with choice: '{choice}'")
            return True
    
    # Additional semantic matching for common variations - more conservative
    semantic_matches = {
        'work': ['office', 'workplace', 'job'],
        'office': ['work', 'workplace', 'desk'],
        'outside': ['outdoor', 'outdoors', 'exterior'],
        'home': ['house', 'residence'],
        'restaurant': ['cafe', 'diner', 'eatery'],
    }
    
    for key, synonyms in semantic_matches.items():
        if key == predicted_lower:  # Only exact semantic matches
            # Check if any synonyms match the direct answers or choices
            try:
                answer_list = ast.literal_eval(direct_answers)
                for answer in answer_list:
                    if any(syn == answer.lower().strip() for syn in synonyms):
                        print(f"    SEMANTIC MATCH found: '{key}' -> '{answer}'")
                        return True
            except:
                pass
            
            for choice in choices:
                if any(syn == choice.lower().strip() for syn in synonyms):
                    print(f"    SEMANTIC MATCH found: '{key}' -> '{choice}'")
                    return True
    
    print(f"    NO MATCH found")
    return False

# ======================================
# 5. PREPARE EXCEL WORKBOOK (incremental writing)
# ======================================
wb = Workbook()
ws_examples = wb.active
ws_examples.title = "per_example"

# header row for per-example sheet
ws_examples.append([
    "question_id",
    "question",
    "num_choices",
    "choices",  # Store all choices as a single field
    "correct_choice_idx",
    "direct_answers",
    "predicted_answer",
    "correct",
    "inference_mode",
    "raw_output",
])

# create summary sheet
ws_summary = wb.create_sheet("summary")
ws_summary.append([
    "metric",
    "value",
])

# save initial empty file
wb.save(OUTPUT_XLSX)

# ======================================
# 6. MAIN EVAL LOOP FOR OKVQA
# ======================================
overall_correct = 0
overall_total = 0

print(f"\n=== Evaluating {NUM_QUESTIONS} A-OKVQA questions using {INFERENCE_MODE} inference ===")

for i, idx in enumerate(selected_indices):
    row = data_split[idx]

    question_id = row["question_id"]
    question = row["question"]
    choices = row["choices"]
    correct_choice_idx = row["correct_choice_idx"]
    direct_answers = row["direct_answers"]
    image = row["image"]  # Get the image from the dataset

    # Build prompt based on inference mode
    if INFERENCE_MODE == "0-shot":
        prompt = build_prompt(question, choices)
    elif INFERENCE_MODE == "4-shot":
        prompt = build_prompt(question, choices, few_shot_examples)
    else:
        raise ValueError(f"Unsupported inference mode: {INFERENCE_MODE}")
    
    # Pass both prompt and image to VILA
    output_text = run_vila_infer(prompt, image)
    predicted_answer = extract_answer(output_text, choices)

    is_correct = int(check_answer_correct(predicted_answer, direct_answers, choices))
    overall_correct += is_correct
    overall_total += 1

    print(
        f"  Q{i+1}/{NUM_QUESTIONS}: pred='{predicted_answer}', "
        f"correct={is_correct}"
    )

    # append example row to Excel and save immediately
    ws_examples.append([
        question_id,
        question,
        len(choices),
        " | ".join(choices),  # Join all choices with separator
        correct_choice_idx,
        direct_answers,
        predicted_answer,
        is_correct,
        INFERENCE_MODE,
        output_text,
    ])
    wb.save(OUTPUT_XLSX)

# ======================================
# 7. OVERALL STATS + FINAL SAVE
# ======================================
if overall_total > 0:
    overall_accuracy = overall_correct / overall_total * 100.0
else:
    overall_accuracy = 0.0

print("\n===========================================")
print(f"A-OKVQA Evaluation Results ({INFERENCE_MODE}):")
print(f"Total questions: {overall_total}")
print(f"Correct answers: {overall_correct}")
print(f"Accuracy: {overall_accuracy:.2f}%")
print("===========================================")

# add summary statistics
ws_summary.append(["Total Questions", overall_total])
ws_summary.append(["Correct Answers", overall_correct])
ws_summary.append(["Accuracy (%)", f"{overall_accuracy:.2f}"])
ws_summary.append(["Inference Mode", INFERENCE_MODE])
ws_summary.append(["Model Path", MODEL_PATH])
ws_summary.append(["Dataset", "HuggingFaceM4/A-OKVQA"])
ws_summary.append(["Split", SPLIT])

if INFERENCE_MODE == "4-shot":
    ws_summary.append(["Number of Shots", NUM_SHOTS])

# Add choice distribution statistics
choice_counts = {}
for idx in selected_indices:
    row = data_split[idx]
    num_choices = len(row["choices"])
    choice_counts[num_choices] = choice_counts.get(num_choices, 0) + 1

ws_summary.append(["--- Choice Distribution ---", ""])
for num, count in sorted(choice_counts.items()):
    ws_summary.append([f"Questions with {num} choices", count])
wb.save(OUTPUT_XLSX)

print(f"\nResults saved incrementally to: {OUTPUT_XLSX}")
print("Done.")

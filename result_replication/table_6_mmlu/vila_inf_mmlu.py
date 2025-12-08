import random
import re
import subprocess
from typing import List

from datasets import load_dataset
from openpyxl import Workbook

# ======================================
# CONFIG
# ======================================
MODEL_PATH = "Efficient-Large-Model/VILA1.5-7b"
CONV_MODE = "vicuna_v1"
NUM_SUBJECTS = 10                 # <---- number of random classes
NUM_QUESTIONS_PER_SUBJECT = 10    # <---- questions per class
SPLIT = "test"                    # or "validation" if you prefer
OUTPUT_XLSX = "vila_mmlu_10x10.xlsx"
SEED = 42

random.seed(SEED)

# ======================================
# 1. LOAD DATASET (cais/mmlu, "all")
# ======================================
print("Loading cais/mmlu (config='all')...")
dataset = load_dataset("cais/mmlu", "all")

print("Available splits:", dataset.keys())
data_split = dataset[SPLIT]
subjects = list(data_split.unique("subject"))
print(f"Found {len(subjects)} subjects in split '{SPLIT}'.")

# pick 10 random subjects
if NUM_SUBJECTS > len(subjects):
    NUM_SUBJECTS = len(subjects)
selected_subjects = random.sample(subjects, NUM_SUBJECTS)
print(f"\nSelected {NUM_SUBJECTS} random subjects:")
for s in selected_subjects:
    print("  -", s)

# ======================================
# 2. PROMPT BUILDER
# ======================================
def build_prompt(question: str, choices: List[str]) -> str:
    """
    Build a multiple-choice prompt for MMLU.
    We explicitly tell the model to answer with only A/B/C/D.
    """
    letters = ["A", "B", "C", "D"]
    prompt = f"Question: {question}\n"
    for L, choice in zip(letters, choices):
        prompt += f"{L}. {choice}\n"
    prompt += (
        "\nPlease answer with only the letter of the correct option "
        "(A, B, C, or D).\nAnswer:"
    )
    return prompt

# ======================================
# 3. CALL VILA VIA CLI (TEXT-ONLY)
# ======================================
def run_vila_infer(prompt: str) -> str:
    """
    Call the VILA CLI in text-only mode.
    Assumes `vila-infer` is on PATH.
    """
    cmd = [
        "vila-infer",
        "--model-path", MODEL_PATH,
        "--conv-mode", CONV_MODE,
        "--text", prompt,
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

# ======================================
# 4. EXTRACT ANSWER LETTER
# ======================================
def extract_answer_letter(output_text: str) -> str:
    """
    From VILA's output text, extract the chosen option letter (A/B/C/D).
    Because we ask for only the letter, often the last non-empty line is it.
    """
    if not output_text.strip():
        return ""

    # Try last non-empty line first
    lines = [l.strip() for l in output_text.splitlines() if l.strip()]
    if lines:
        last = lines[-1]
        m = re.search(r"\b([A-D])\b", last)
        if m:
            return m.group(1)

    # Fallback: search anywhere
    m = re.search(r"\b([A-D])\b", output_text)
    if m:
        return m.group(1)

    return ""

# ======================================
# 5. PREPARE EXCEL WORKBOOK (incremental writing)
# ======================================
wb = Workbook()
ws_examples = wb.active
ws_examples.title = "per_example"

# header row for per-example sheet
ws_examples.append([
    "subject",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "gold_letter",
    "pred_letter",
    "correct",
    "raw_output",
])

# create per_subject sheet
ws_subjects = wb.create_sheet("per_subject")
ws_subjects.append([
    "subject",
    "num_questions",
    "num_correct",
    "accuracy_percent",
])

# save initial empty file
wb.save(OUTPUT_XLSX)

# ======================================
# 6. MAIN EVAL LOOP OVER SELECTED SUBJECTS
# ======================================
letters = ["A", "B", "C", "D"]
overall_correct = 0
overall_total = 0

for subject in selected_subjects:
    print(f"\n=== Subject '{subject}' ===")

    # Filter this split for the current subject
    subj_data = data_split.filter(lambda ex: ex["subject"] == subject)
    total_available = len(subj_data)
    if total_available == 0:
        print(f"  No data for subject {subject}, skipping.")
        continue

    n_use = min(NUM_QUESTIONS_PER_SUBJECT, total_available)
    indices = random.sample(range(total_available), n_use)
    subj_correct = 0

    print(f"  Using {n_use} questions out of {total_available} available.")

    for i, idx in enumerate(indices):
        row = subj_data[idx]

        question = row["question"]
        choices = row["choices"]
        gold_index = row["answer"]          # 0–3
        gold_letter = letters[gold_index]   # A–D

        prompt = build_prompt(question, choices)
        output_text = run_vila_infer(prompt)
        pred_letter = extract_answer_letter(output_text)

        is_correct = int(pred_letter == gold_letter)
        subj_correct += is_correct
        overall_correct += is_correct
        overall_total += 1

        print(
            f"  Q{i+1}/{n_use}: gold={gold_letter}, "
            f"pred={pred_letter or '<empty>'}, correct={is_correct}"
        )

        # append example row to Excel and save immediately
        ws_examples.append([
            subject,
            question,
            choices[0],
            choices[1],
            choices[2],
            choices[3],
            gold_letter,
            pred_letter,
            is_correct,
            output_text,
        ])
        wb.save(OUTPUT_XLSX)

    accuracy = subj_correct / n_use * 100.0
    print(f"  Subject '{subject}' accuracy: {accuracy:.2f}%")

    # append subject summary row and save
    ws_subjects.append([
        subject,
        n_use,
        subj_correct,
        accuracy,
    ])
    wb.save(OUTPUT_XLSX)

# ======================================
# 7. OVERALL STATS + FINAL SAVE
# ======================================
if overall_total > 0:
    overall_accuracy = overall_correct / overall_total * 100.0
else:
    overall_accuracy = 0.0

print("\n==========================================")
print(f"Overall accuracy (sampled): {overall_accuracy:.2f}%")
print("==========================================")

# add overall row to per_subject sheet
ws_subjects.append([
    "__OVERALL__",
    overall_total,
    overall_correct,
    overall_accuracy,
])
wb.save(OUTPUT_XLSX)

print(f"\nResults saved incrementally to: {OUTPUT_XLSX}")
print("Done.")

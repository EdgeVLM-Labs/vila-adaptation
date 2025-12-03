import json
import csv
import subprocess
import time
import psutil
import os

PROMPT = "Please evaluate the exercise form shown. What mistakes, if any, are present, and what corrections would you recommend?"

def run_inference(model_path, conv_mode, text, media):
    start_time = time.time()
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss / (1024 * 1024)  # in MB

    # Run the vila-infer command and capture output
    result = subprocess.run([
        "vila-infer",
        "--model-path", model_path,
        "--conv-mode", conv_mode,
        "--text", text,
        "--media", media
    ], capture_output=True, text=True)

    mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
    end_time = time.time()
    inference_time = end_time - start_time
    memory_usage = mem_after - mem_before

    # Get model output from stdout
    model_output = result.stdout.strip()
    return model_output, memory_usage, inference_time

def main(json_path, csv_path, model_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    import sys
    limit = None
    if len(sys.argv) == 5:
        try:
            limit = int(sys.argv[4])
        except ValueError:
            print("Invalid limit argument, must be an integer.")
            sys.exit(1)

    if limit is not None:
        data = data[:limit]

    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["video", "ground_truth", "model_output", "memory_usage_mb", "inference_time_sec"])

        for entry in data:
            video = entry.get("video", "")
            ground_truth = ""
            # Extract ground truth from conversations
            for conv in entry.get("conversations", []):
                if conv.get("from") == "gpt":
                    ground_truth = conv.get("value", "")
                    break

            # Run inference
            model_output, memory_usage, inference_time = run_inference(
                model_path,
                "vicuna_v1",
                PROMPT,
                video
            )

            writer.writerow([video, ground_truth, model_output, f"{memory_usage:.2f}", f"{inference_time:.2f}"])
            print(f"Processed {video}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [4, 5]:
        print("Usage: python inference_script.py <input_json> <output_csv> <model_path> [limit]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])   #model_output, memory_usage, inference_time


def main(json_path, csv_path, model_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    import sys
    limit = None
    if len(sys.argv) == 5:
        try:
            limit = int(sys.argv[4])
        except ValueError:
            print("Invalid limit argument, must be an integer.")
            sys.exit(1)

    if limit is not None:
        data = data[:limit]

    import os
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline='') as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            writer.writerow(["video", "ground_truth", "model_output", "memory_usage_mb", "inference_time_sec"])

        for entry in data:
            video = entry.get("video", "")
            ground_truth = ""
            # Extract ground truth from conversations
            for conv in entry.get("conversations", []):
                if conv.get("from") == "gpt":
                    ground_truth = conv.get("value", "")
                    break

            # Run inference
            model_output, memory_usage, inference_time = run_inference(
                model_path,
                "vicuna_v1",
                PROMPT,
                video
            )

            writer.writerow([video, ground_truth, model_output, f"{memory_usage:.2f}", f"{inference_time:.2f}"])
            print(f"Processed {video}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) not in [4, 5]:
        print("Usage: python inference_script.py <input_json> <output_csv> <model_path> [limit]")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2], sys.argv[3])
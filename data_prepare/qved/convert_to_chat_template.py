import json
import sys

def convert_to_dataset_format(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    output = []
    for item in data:
        video = item.get("video", item.get("video_path", ""))  # Use "video" or "image" key
        if video.startswith("./"):
            video = video[2:]
        labels = item.get("labels_descriptive", [])
        gpt_value = " ".join(labels) if isinstance(labels, list) else str(labels)
        entry = {
            "video": "./llava/data/registry/datasets/dataset/videos/" + video,
            "conversations": [
                {
                    "from": "human",
                    "value": "Give some feedback about this exercise"
                },
                {
                    "from": "gpt",
                    "value": gpt_value
                }
            ]
        }
        output.append(entry)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert.py <input_json> <output_json>")
        sys.exit(1)
    convert_to_dataset_format(sys.argv[1], sys.argv[2])
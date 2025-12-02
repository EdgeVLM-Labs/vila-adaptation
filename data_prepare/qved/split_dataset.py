import json
import random
import sys

def split_dataset(input_path, train_path, test_path, train_ratio=0.8):
    with open(input_path, 'r') as f:
        data = json.load(f)

    random.shuffle(data)
    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=4)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python split_dataset.py <input_json> <train_json> <test_json>")
        sys.exit(1)
    split_dataset(sys.argv[1], sys.argv[2], sys.argv[3])
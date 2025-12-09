import json
import random
import sys

def split_dataset(input_path, train_path, val_path, test_path, train_ratio=0.7, val_ratio=0.15):
    """
    Split dataset into train/validation/test sets.
    Default: 70% train, 15% validation, 15% test
    """
    with open(input_path, 'r') as f:
        data = json.load(f)

    random.seed(42)  # Ensures consistent splits every time
    random.shuffle(data)
    
    # Calculate split indices
    train_idx = int(len(data) * train_ratio)
    val_idx = int(len(data) * (train_ratio + val_ratio))
    
    # Split the data
    train_data = data[:train_idx]
    val_data = data[train_idx:val_idx]
    test_data = data[val_idx:]

    # Save splits
    with open(train_path, 'w') as f:
        json.dump(train_data, f, indent=4)
    with open(val_path, 'w') as f:
        json.dump(val_data, f, indent=4)
    with open(test_path, 'w') as f:
        json.dump(test_data, f, indent=4)
    
    print(f"Dataset split completed:")
    print(f"  Total samples: {len(data)}")
    print(f"  Train: {len(train_data)} ({len(train_data)/len(data)*100:.1f}%)")
    print(f"  Validation: {len(val_data)} ({len(val_data)/len(data)*100:.1f}%)")
    print(f"  Test: {len(test_data)} ({len(test_data)/len(data)*100:.1f}%)")

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python split_dataset.py <input_json> <train_json> <val_json> <test_json>")
        print("Example: python split_dataset.py dataset.json train.json val.json test.json")
        sys.exit(1)
    split_dataset(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
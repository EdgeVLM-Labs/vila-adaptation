#!/usr/bin/env python3
"""
Video Augmentation Script for QVED Dataset
Uses vidaug library to apply various augmentation techniques to exercise videos.
"""

import sys

# Check for required dependencies
try:
    import skimage
except ImportError:
    print("Error: scikit-image is not installed.")
    print("Please install it with: pip install scikit-image")
    sys.exit(1)

try:
    import vidaug.augmentors as va
except ImportError:
    print("Error: vidaug is not installed.")
    print("Please install it with: pip install vidaug")
    sys.exit(1)

import json
import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter

# Base directory - dataset is at /workspace/vila-adaptation/llava/data/registry/datasets/dataset
BASE_DIR = Path("/workspace/vila-adaptation/llava/data/registry/datasets/dataset")
VIDEOS_DIR = BASE_DIR / "videos"
GROUND_TRUTH_FILE = Path("/workspace/vila-adaptation/llava/data/registry/datasets/fine_grained_labels.json")
MANIFEST_FILE = Path("/workspace/vila-adaptation/llava/data/registry/datasets/manifest.json")
OUTPUT_GROUND_TRUTH_FILE = Path("/workspace/vila-adaptation/llava/data/registry/datasets/ground_truth.json")

# Define available augmentations with numbers
AUGMENTATION_OPTIONS = {
    1: ("Horizontal Flip", va.HorizontalFlip()),
    2: ("Vertical Flip", va.VerticalFlip()),
    3: ("Random Rotate (±10°)", va.RandomRotate(degrees=10)),
    4: ("Random Resize (±20%)", va.RandomResize(rate=0.2)),
    5: ("Gaussian Blur", va.GaussianBlur(sigma=1.5)),
    6: ("Add Brightness (+30)", va.Add(value=30)),
    7: ("Multiply Brightness (1.2x)", va.Multiply(value=1.2)),
    8: ("Random Translate (±15px)", va.RandomTranslate(x=15, y=15)),
    9: ("Random Shear", va.RandomShear(x=0.1, y=0.1)),
    10: ("Invert Color", va.InvertColor()),
    11: ("Salt Noise", va.Salt(ratio=100)),
    12: ("Pepper Noise", va.Pepper(ratio=100)),
    13: ("Temporal Downsample (0.8x)", va.Downsample(ratio=0.8)),
    14: ("Elastic Transformation", va.ElasticTransformation(alpha=10, sigma=3)),
}


def display_augmentation_options():
    """Display all available augmentation options."""
    print("\n" + "="*60)
    print("Available Video Augmentation Techniques:")
    print("="*60)
    for idx, (name, _) in AUGMENTATION_OPTIONS.items():
        print(f"  {idx:2d}. {name}")
    print("="*60 + "\n")


def load_video_frames(video_path):
    """Load video frames as a list of PIL Images."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        frames.append(pil_image)

    cap.release()
    return frames


def save_video_frames(frames, output_path, fps=30):
    """Save frames as a video file."""
    if not frames:
        print(f"Warning: No frames to save for {output_path}")
        return False

    # Convert first frame to get dimensions
    first_frame = np.array(frames[0])
    height, width = first_frame.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for frame in frames:
        # Convert PIL Image to numpy array
        frame_np = np.array(frame)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    return True


def get_video_fps(video_path):
    """Get the FPS of a video."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30


def augment_video(video_path, augmentors, output_path):
    """Apply augmentation to a video."""
    print(f"  Processing: {video_path.name}...", end=" ", flush=True)

    # Load video frames
    frames = load_video_frames(video_path)

    if not frames:
        print("❌ Failed to load frames")
        return False

    # Apply augmentation sequence
    try:
        augmented_frames = augmentors(frames)
    except Exception as e:
        print(f"❌ Augmentation failed: {e}")
        return False

    # Get original FPS
    fps = get_video_fps(video_path)

    # Save augmented video
    success = save_video_frames(augmented_frames, output_path, fps)

    if success:
        print("✓")
        return True
    else:
        print("❌ Failed to save")
        return False


def update_json_files(augmented_videos_info):
    """Update JSON files with augmented video paths."""
    print("\n" + "="*60)
    print("Updating JSON files with augmented videos...")
    print("="*60)

    # Load existing JSON files
    with open(GROUND_TRUTH_FILE, 'r') as f:
        fine_grained_labels = json.load(f)

    with open(MANIFEST_FILE, 'r') as f:
        manifest = json.load(f)

    # Load ground_truth.json if it exists
    ground_truth = {}
    if OUTPUT_GROUND_TRUTH_FILE.exists():
        with open(OUTPUT_GROUND_TRUTH_FILE, 'r') as f:
            ground_truth = json.load(f)

    # Add augmented videos to JSON files
    for aug_info in augmented_videos_info:
        original_path = aug_info['original_path']
        augmented_path = aug_info['augmented_path']

        # Find original entry in fine_grained_labels
        if original_path in fine_grained_labels:
            # Copy the entry for augmented video
            fine_grained_labels[augmented_path] = fine_grained_labels[original_path].copy()

        # Add to manifest (copy from original if exists)
        # Use string format to maintain consistency with original entries
        if original_path in manifest:
            # If original is a string, use it directly; if dict, extract the exercise name
            original_value = manifest[original_path]
            if isinstance(original_value, str):
                manifest[augmented_path] = original_value
            elif isinstance(original_value, dict) and 'path' in original_value:
                # Get exercise name from original entry's key
                manifest[augmented_path] = original_value.get('exercise', original_value.get('path', augmented_path).split('/')[0].replace('_', ' '))
            else:
                manifest[augmented_path] = augmented_path.split('/')[0].replace('_', ' ')
        else:
            # Extract exercise name from path (folder name)
            exercise_name = augmented_path.split('/')[0]
            manifest[augmented_path] = exercise_name

        # Add to ground_truth.json if original exists there
        if original_path in ground_truth:
            ground_truth[augmented_path] = ground_truth[original_path].copy()

    # Save updated JSON files
    with open(GROUND_TRUTH_FILE, 'w') as f:
        json.dump(fine_grained_labels, f, indent=2)
    print(f"✓ Updated {GROUND_TRUTH_FILE}")

    with open(MANIFEST_FILE, 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"✓ Updated {MANIFEST_FILE}")

    if ground_truth:
        with open(OUTPUT_GROUND_TRUTH_FILE, 'w') as f:
            json.dump(ground_truth, f, indent=2)
        print(f"✓ Updated {OUTPUT_GROUND_TRUTH_FILE}")

    print(f"\n✓ Added {len(augmented_videos_info)} augmented videos to JSON files")


def main():
    print("\n" + "="*60)
    print("Video Augmentation Tool for QVED Dataset")
    print("="*60)

    # Check if dataset directory exists
    if not BASE_DIR.exists():
        print(f"❌ Error: Dataset directory not found at {BASE_DIR}")
        print(f"   Current script location: {Path(__file__).resolve()}")
        print(f"   Looking for dataset at: {BASE_DIR.resolve()}")
        print("\nPlease ensure the dataset directory exists or update BASE_DIR in the script.")
        sys.exit(1)

    # Check if videos directory exists
    if not VIDEOS_DIR.exists():
        print(f"❌ Error: Videos directory not found at {VIDEOS_DIR}")
        print("\nPlease ensure the videos directory exists.")
        sys.exit(1)

    # Check if videos are directly in VIDEOS_DIR or in subfolders
    direct_videos = sorted(list(VIDEOS_DIR.glob("*.mp4")))
    exercise_folders = sorted([d for d in VIDEOS_DIR.iterdir() if d.is_dir() and list(d.glob("*.mp4"))])

    if direct_videos:
        # Videos are directly in the videos/ folder (flat structure)
        print(f"\n✓ Found {len(direct_videos)} videos directly in videos directory")
        print("   Using flat structure (all videos in videos/ folder)")
        
        # Group all videos under a single "all_videos" category
        videos_by_category = {"all_videos": direct_videos}
        
    elif exercise_folders:
        # Videos are organized in exercise subfolders
        print(f"\n✓ Found {len(exercise_folders)} exercise folders")
        videos_by_category = {}
        
        print("\nExercise folders and video counts:")
        print("-" * 60)
        for idx, folder in enumerate(exercise_folders, 1):
            videos = sorted(list(folder.glob("*.mp4")))
            videos_by_category[folder.name] = videos
            print(f"  {idx}. {folder.name:<40} ({len(videos)} videos)")
        print("-" * 60)
    else:
        print("❌ No videos found in videos directory!")
        print(f"   Searched in: {VIDEOS_DIR.resolve()}")
        return

    # Display augmentation options
    display_augmentation_options()

    # Ask which videos to augment
    if len(videos_by_category) == 1 and "all_videos" in videos_by_category:
        # Flat structure - ask how many videos to augment
        total_videos = len(videos_by_category["all_videos"])
        print(f"\nFound {total_videos} videos in flat structure")
        print("Enter number of videos to augment (or 'all' for all videos):")
        count_input = input("Number of videos: ").strip().lower()
        
        if count_input == 'all':
            selected_videos = videos_by_category["all_videos"]
        else:
            try:
                count = int(count_input)
                selected_videos = videos_by_category["all_videos"][:count]
            except ValueError:
                print("❌ Invalid input!")
                return
        
        videos_to_process = {"all_videos": selected_videos}
        
    else:
        # Folder structure - let user choose folders
        folder_names = list(videos_by_category.keys())
        print("\nEnter the indices of folders you want to augment (comma-separated)")
        print("Example: 1,3,5 or just press Enter to augment all")
        folder_input = input("Folder indices: ").strip()

        if folder_input:
            try:
                selected_indices = [int(x.strip()) for x in folder_input.split(',')]
                videos_to_process = {
                    folder_names[i-1]: videos_by_category[folder_names[i-1]] 
                    for i in selected_indices if 1 <= i <= len(folder_names)
                }
            except (ValueError, IndexError):
                print("❌ Invalid input! Please enter valid comma-separated numbers.")
                return
        else:
            videos_to_process = videos_by_category

        print(f"\n✓ Selected {len(videos_to_process)} folder(s) for augmentation")

    # Track all augmented videos for JSON update
    all_augmented_videos = []

    # For each selected category/folder, ask for augmentation techniques
    for category_name, videos in videos_to_process.items():
        print("\n" + "="*60)
        if category_name == "all_videos":
            print(f"Processing {len(videos)} videos from flat structure")
        else:
            print(f"Folder: {category_name}")
        print("="*60)

        if not videos:
            print("No videos found, skipping...")
            continue

        print(f"Found {len(videos)} video(s)")
        print("\nEnter augmentation techniques to apply (comma-separated indices)")
        print("Example: 1,3,5 for Horizontal Flip, Random Rotate, Gaussian Blur")
        aug_input = input("Augmentation indices: ").strip()

        if not aug_input:
            print("No augmentations selected, skipping...")
            continue

        try:
            selected_aug_indices = [int(x.strip()) for x in aug_input.split(',')]
            selected_augmentors = []
            aug_names = []

            for idx in selected_aug_indices:
                if idx in AUGMENTATION_OPTIONS:
                    name, augmentor = AUGMENTATION_OPTIONS[idx]
                    selected_augmentors.append(augmentor)
                    aug_names.append(name)
                else:
                    print(f"Warning: Invalid augmentation index {idx}, skipping...")

            if not selected_augmentors:
                print("No valid augmentations selected, skipping...")
                continue

            print(f"\n✓ Will apply: {', '.join(aug_names)}")

            # Process each video
            for video_path in videos:
                # Generate output filename with augmentation index
                for aug_idx in selected_aug_indices:
                    if aug_idx not in AUGMENTATION_OPTIONS:
                        continue

                    # Create individual augmentor for this technique
                    _, augmentor = AUGMENTATION_OPTIONS[aug_idx]
                    single_aug = va.Sequential([augmentor])

                    video_stem = video_path.stem
                    output_filename = f"{video_stem}_aug{aug_idx}.mp4"
                    
                    # Save in the same directory as the original
                    output_path = video_path.parent / output_filename

                    # Apply augmentation
                    success = augment_video(video_path, single_aug, output_path)

                    if success:
                        # Track for JSON update
                        if category_name == "all_videos":
                            # Flat structure: videos/filename.mp4
                            json_original_path = f"videos/{video_path.name}"
                            json_augmented_path = f"videos/{output_filename}"
                        else:
                            # Folder structure: videos/folder/filename.mp4
                            json_original_path = f"videos/{category_name}/{video_path.name}"
                            json_augmented_path = f"videos/{category_name}/{output_filename}"
                        
                        all_augmented_videos.append({
                            'original_path': json_original_path,
                            'augmented_path': json_augmented_path
                        })

        except ValueError:
            print("❌ Invalid input! Please enter valid comma-separated numbers.")
            continue

    # Update JSON files
    if all_augmented_videos:
        update_json_files(all_augmented_videos)
        print("\n" + "="*60)
        print(f"✓ Augmentation Complete! Created {len(all_augmented_videos)} augmented videos")
        print("="*60)
    else:
        print("\n❌ No videos were augmented")


if __name__ == "__main__":
    main()
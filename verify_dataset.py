#!/usr/bin/env python3
"""
Verify VILA video dataset format
Usage: python verify_dataset.py data/my_video_sft/train.jsonl data/my_video_sft/videos
"""

import json
import os
import sys
from pathlib import Path


def verify_dataset(jsonl_path: str, video_dir: str):
    """Verify dataset format and video files exist."""
    
    print("=" * 60)
    print("VILA Video Dataset Verification")
    print("=" * 60)
    
    # Check JSONL file exists
    if not os.path.exists(jsonl_path):
        print(f"❌ ERROR: JSONL file not found: {jsonl_path}")
        return False
    
    print(f"✓ JSONL file found: {jsonl_path}")
    
    # Check video directory exists
    if not os.path.exists(video_dir):
        print(f"❌ ERROR: Video directory not found: {video_dir}")
        return False
    
    print(f"✓ Video directory found: {video_dir}")
    print()
    
    # Parse JSONL
    entries = []
    errors = []
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
                
            try:
                entry = json.loads(line)
                entries.append((line_num, entry))
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
    
    print(f"📊 Found {len(entries)} entries in dataset")
    print()
    
    # Validate each entry
    missing_videos = []
    format_errors = []
    valid_count = 0
    
    for line_num, entry in entries:
        # Check required fields
        if 'id' not in entry:
            format_errors.append(f"Line {line_num}: Missing 'id' field")
            continue
            
        if 'video' not in entry:
            format_errors.append(f"Line {line_num}: Missing 'video' field")
            continue
            
        if 'conversations' not in entry:
            format_errors.append(f"Line {line_num}: Missing 'conversations' field")
            continue
        
        # Check conversations format
        convs = entry['conversations']
        if not isinstance(convs, list) or len(convs) < 2:
            format_errors.append(f"Line {line_num}: 'conversations' must be a list with at least 2 entries")
            continue
        
        # Validate conversation structure
        for i, conv in enumerate(convs):
            if 'from' not in conv or 'value' not in conv:
                format_errors.append(f"Line {line_num}, Conv {i}: Missing 'from' or 'value' field")
                continue
            
            if conv['from'] not in ['human', 'gpt']:
                format_errors.append(f"Line {line_num}, Conv {i}: 'from' must be 'human' or 'gpt'")
                continue
        
        # Check if video file exists (add .mp4 extension)
        video_path = os.path.join(video_dir, entry['video'] + '.mp4')
        if not os.path.exists(video_path):
            missing_videos.append(f"Line {line_num}: Video not found: {video_path}")
        else:
            valid_count += 1
    
    # Print results
    print("=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    
    if errors:
        print(f"\n❌ JSON Parsing Errors ({len(errors)}):")
        for error in errors[:10]:  # Show first 10
            print(f"  - {error}")
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more")
    
    if format_errors:
        print(f"\n❌ Format Errors ({len(format_errors)}):")
        for error in format_errors[:10]:
            print(f"  - {error}")
        if len(format_errors) > 10:
            print(f"  ... and {len(format_errors) - 10} more")
    
    if missing_videos:
        print(f"\n⚠️  Missing Videos ({len(missing_videos)}):")
        for video in missing_videos[:10]:
            print(f"  - {video}")
        if len(missing_videos) > 10:
            print(f"  ... and {len(missing_videos) - 10} more")
    
    print(f"\n✓ Valid entries: {valid_count}/{len(entries)}")
    
    if valid_count == len(entries) and not errors:
        print("\n✅ Dataset is valid and ready for training!")
        return True
    else:
        print("\n❌ Dataset has issues that need to be fixed")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python verify_dataset.py <path_to_jsonl> <path_to_videos>")
        print("Example: python verify_dataset.py data/my_video_sft/train.jsonl data/my_video_sft/videos")
        sys.exit(1)
    
    jsonl_path = sys.argv[1]
    video_dir = sys.argv[2]
    
    success = verify_dataset(jsonl_path, video_dir)
    sys.exit(0 if success else 1)

import os
import shutil

def copy_videos_to_folder(source_root, target_videos_dir):
    os.makedirs(target_videos_dir, exist_ok=True)
    for subdir in os.listdir(source_root):
        subdir_path = os.path.join(source_root, subdir)
        # Skip the target videos directory to avoid SameFileError
        if subdir == os.path.basename(target_videos_dir):
            continue
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.mp4'):
                    src_file = os.path.join(subdir_path, file)
                    dst_file = os.path.join(target_videos_dir, file)
                    shutil.copy2(src_file, dst_file)
                    print(f"Copied {src_file} to {dst_file}")

        
if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python split_dataset.py <source_root> <target_videos_dir>")
        sys.exit(1)
    copy_videos_to_folder(sys.argv[1], sys.argv[2])
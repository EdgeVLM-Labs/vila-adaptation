#!/usr/bin/env python3
import os
import glob

# Find all Python files with the problematic import
files_to_fix = []
for pattern in [
    'llava/model/**/*.py',
]:
    files_to_fix.extend(glob.glob(pattern, recursive=True))

fixed_count = 0
for filepath in files_to_fix:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Check if file has the problematic import and is not already fixed
        if 'from triton.language.extra.cuda import libdevice' in content:
            if 'try:\n    from triton.language.extra.cuda import libdevice' not in content:
                # Fix the import
                new_content = content.replace(
                    'from triton.language.extra.cuda import libdevice',
                    'try:\n    from triton.language.extra.cuda import libdevice\nexcept ImportError:\n    libdevice = None'
                )
                
                with open(filepath, 'w') as f:
                    f.write(new_content)
                print(f"Fixed: {filepath}")
                fixed_count += 1
    except Exception as e:
        print(f"Error processing {filepath}: {e}")

print(f"\nTotal files fixed: {fixed_count}")

# Clean Python cache
print("Cleaning Python cache...")
os.system("find llava -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null")
os.system("find llava -type f -name '*.pyc' -delete 2>/dev/null")
print("Done!")

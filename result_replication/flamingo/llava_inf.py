import requests
from PIL import Image
import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration

# 1. SETUP MODEL
model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
).to(0)
processor = AutoProcessor.from_pretrained(model_id)

# 2. LOAD & STITCH IMAGES (The "Hack" to bypass the crash)
# Ensure these files exist locally
image_paths = ["../data/img1.png", "../data/img2.png", "../data/img3.png"] 
raw_images = [Image.open(p) for p in image_paths]

def stitch_images(images):
    # Combine images side-by-side
    total_width = sum(img.width for img in images)
    max_height = max(img.height for img in images)
    new_img = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for img in images:
        new_img.paste(img, (x_offset, 0))
        x_offset += img.width
    return new_img

stitched_image = stitch_images(raw_images)

# 3. CONSTRUCT CONVERSATION
# We use a single <image> token because we stitched them.
# This mimics the "confusion" state: the model sees all 3 flamingos but can't separate them.
conversation = [
    {
      "role": "user",
      "content": [
          {"type": "text", "text": "Image 1 is on the left. Image 2 is in the middle. Image 3 is on the right. What is the common thing about these three images? Also describe the different artistic style of each image."},
          {"type": "image"}, 
      ],
    },
]

# 4. GENERATE
prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
inputs = processor(images=stitched_image, text=prompt, return_tensors='pt').to(0, torch.float16)

output = model.generate(**inputs, max_new_tokens=200, do_sample=False)
print(processor.decode(output[0], skip_special_tokens=True))
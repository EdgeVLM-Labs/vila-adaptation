import torch
from PIL import Image
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token

# -----------------------------
# CONFIG
# -----------------------------
MODEL_PATH = "liuhaotian/llava-v1.5-7b"
MODEL_BASE = None
MODEL_NAME = "llava-v1.5-7b"
IMG1 = "demo_images/underground.png"
IMG2 = "demo_images/congress.png"
IMG3 = "demo_images/soulomes.png"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("Using model: LLaVA 1.5 7B")

# -----------------------------
# LOAD MODEL + TOKENIZER
# -----------------------------
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=MODEL_PATH,
    model_base=MODEL_BASE,
    model_name=MODEL_NAME,
)
model.to(device)

# -----------------------------
# LOAD IMAGES
# -----------------------------
images = [
    Image.open(IMG1).convert("RGB"),
    Image.open(IMG2).convert("RGB"),
    Image.open(IMG3).convert("RGB"),
]
image_tensors = process_images(images, image_processor, model.config)
image_tensors = image_tensors.to(device=device, dtype=torch.float16)

# -----------------------------
# BUILD PROMPT
# -----------------------------
prompt = (
    "Image 1: <image>\nAnswer 1: Underground.\n\n"
    "Image 2: <image>\nAnswer 2: Congress.\n\n"
    "Image 3: <image>\nAnswer 3:"
)
num_image_tokens = prompt.count("<image>")
print("Number of <image> tokens in prompt:", num_image_tokens)

# -----------------------------
# TOKENIZE PROMPT
# -----------------------------
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    return_tensors="pt",
).unsqueeze(0).to(device)

print(f"Input IDs shape: {input_ids.shape}")
print(f"Input text length: {len(prompt)}")

# -----------------------------
# PREPARE IMAGES (LLaVA uses 'images' parameter)
# -----------------------------
images_for_model = [image_tensors[i] for i in range(num_image_tokens)]

# -----------------------------
# RUN INFERENCE
# -----------------------------
with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        images=images_for_model,  # LLaVA uses 'images' not 'media'
        max_new_tokens=100,
        do_sample=False,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

print(f"Output IDs shape: {output_ids.shape}")

# Decode only the newly generated tokens (exclude the input)
input_token_len = input_ids.shape[1]
generated_ids = output_ids[:, input_token_len:]

output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print("\n===== MODEL OUTPUT =====")
print(output)
print("========================\n")

# Also print full output for debugging
full_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("\n===== FULL OUTPUT (with prompt) =====")
print(full_output)
print("======================================\n")
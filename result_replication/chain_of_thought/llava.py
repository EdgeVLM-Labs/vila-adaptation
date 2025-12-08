import torch
from transformers import AutoProcessor, LlavaForConditionalGeneration
from PIL import Image

model_id = "llava-hf/llava-1.5-7b-hf"

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LlavaForConditionalGeneration.from_pretrained(
    model_id,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    low_cpu_mem_usage=True,
).to(device)

processor = AutoProcessor.from_pretrained(model_id)

# Load your two images
images = [Image.open("photo.png"), Image.open("menu.png")]

# Conversation with two image slots
conversation = [
    {
        "role": "user",
        "content": [
            {"type": "image"},            # <-- photo.png
            {"type": "image"},            # <-- menu.png
            {
                "type": "text",
                "text": (
                    "Photo: <img1>\n"
                    "Menu: <img2>\n"
                    "How much should I pay for all the beer on the table "
                    "according to the price on the menu?"
                ),
            },
        ],
    },
]

prompt = processor.apply_chat_template(
    conversation, add_generation_prompt=True
)

inputs = processor(
    images=images,
    text=prompt,
    return_tensors="pt"
).to(device, dtype=model.dtype)

output_ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=False,
)

# Strip off the prompt tokens before decoding
answer = processor.decode(
    output_ids[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True,
)
print("LLaVA answer:", answer)

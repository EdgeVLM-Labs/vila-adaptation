# MUST be first: registers llava_llama
import llava  

import torch
from peft import PeftModel
from transformers import LlamaTokenizer
from llava.model.language_model.llava_llama import LlavaLlamaModel

# Models
base_model_id = "Efficient-Large-Model/VILA1.5-3b"
lora_id = "EdgeVLM-Labs/VILA1.5-3B-finetuned-500"

# ✅ Correct tokenizer repo
tokenizer_id = "lmsys/vicuna-7b-v1.5"

out_dir = "./merged-vila-3b"

# Load tokenizer (NOT from VILA repo)
tokenizer = LlamaTokenizer.from_pretrained(
    tokenizer_id,
    use_fast=False
)

# Load base VILA model
base_model = LlavaLlamaModel.from_pretrained(
    base_model_id,
    device_map="auto",
    trust_remote_code=True
)

# Attach LoRA
model = PeftModel.from_pretrained(
    base_model,
    lora_id
)

# Merge LoRA into base
merged_model = model.merge_and_unload()

# Save merged model + tokenizer
merged_model.save_pretrained(out_dir)
tokenizer.save_pretrained(out_dir)

print(f"✅ Merged VILA model saved to: {out_dir}")

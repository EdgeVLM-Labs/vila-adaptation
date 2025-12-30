#!/usr/bin/env bash
set -e

# CONDA_ENV=${1:-""}
# if [ -n "$CONDA_ENV" ]; then
#     # This is required to activate conda environment
#     eval "$(conda shell.bash hook)"

#     conda create -n $CONDA_ENV python=3.10.14 -y
#     conda activate $CONDA_ENV
#     # This is optional if you prefer to use built-in nvcc
#     conda install -c nvidia cuda-toolkit -y
# else
#     echo "Skipping conda environment creation. Make sure you have the correct environment activated."
# fi


# echo ">>> Activating environment..."
# conda activate vila

# -------------------------------
# Install dependencies
# -------------------------------

# echo ">>> Installing CUDA Toolkit..."
# conda install -c nvidia cuda-toolkit -y

echo ">>> Installing base dependencies..."
# pip install --upgrade pip

# Enable PEP 660 support
pip install --upgrade pip setuptools

echo ">>> Installing HuggingFace transfer..."
pip install hf_transfer

# Install PS3 Torch
# pip install ps3-torch

echo ">>> Installing FlashAttention2..."
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.8/flash_attn-2.5.8+cu122torch2.3cxx11abiFALSE-cp310-cp310-linux_x86_64.whl

echo ">>> Installing VILA..."
pip install -e ".[train,eval]"

echo ">>> Installing additional dependencies..."
# Quantization requires the newest triton version
pip install triton==3.1.0

# numpy - fix dependency issues
pip install numpy==1.26.4

# Evaluation dependencies
pip install bert-score
pip install wandb

# Downgrade protobuf for backward compatibility
pip install protobuf==3.20.*

# Install PEFT for LoRA support
pip install peft==0.10.0

echo ">>> Installing video processing dependencies..."
# Video augmentation
pip install vidaug
pip install opencv-python-headless
pip install pillow
pip install scikit-image  # Required by vidaug
pip install nltk==3.7
pip install evaluate rouge-score
python -m nltk.downloader wordnet omw-1.4


echo ">>> Installing PS3..."
git clone https://github.com/NVlabs/PS3.git
cd PS3
pip install -e .
cd ..

echo ">>> Installing evaluation and report generation libraries..."
pip install openpyxl
pip install scikit-learn
pip install evaluate
pip install sentence-transformers

echo ">>> Replacing transformers and deepspeed files..."
site_pkg_path=$(python -c 'import site; print(site.getsitepackages()[0])')
cp -rv ./llava/train/deepspeed_replace/* $site_pkg_path/deepspeed/

echo "🔑 Logging into WandB..."
wandb login

echo "🤗 Logging into HuggingFace Hub..."
huggingface-cli login

echo "✅ Environment setup completed successfully!"
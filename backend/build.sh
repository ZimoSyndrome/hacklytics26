#!/bin/bash
source .venv/bin/activate

pip install --upgrade pip

# Install CPU-only PyTorch first
pip install -i https://download.pytorch.org/whl/cpu torch==2.10.0

# Install other packages
pip install flask flask-cors lightgbm numpy scikit-learn joblib pdfplumber

# Install sentence-transformers with CPU PyTorch (no CUDA re-download)
pip install sentence-transformers --no-deps
pip install transformers huggingface-hub safetensors
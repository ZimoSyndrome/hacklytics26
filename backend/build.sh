#!/bin/bash
uv pip install --upgrade pip
uv pip install -i https://download.pytorch.org/whl/cpu torch==2.10.0
uv pip install flask flask-cors lightgbm numpy scikit-learn joblib sentence-transformers pdfplumber
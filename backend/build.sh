#!/bin/bash
uv pip install --upgrade pip
uv pip install -i https://download.pytorch.org/whl/cpu torch==2.0.0
uv pip install flask flask-cors lightgbm numpy scikit-learn joblib pdfplumber --no-cache-dir
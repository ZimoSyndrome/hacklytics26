#!/bin/bash
pip install --upgrade pip
pip install -i https://download.pytorch.org/whl/cpu torch==2.10.0
pip install flask flask-cors lightgbm numpy scikit-learn joblib sentence-transformers pdfplumber
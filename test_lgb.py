import numpy as np
import lightgbm as lgb
import gc
import sys

print("Loading data...")
sys.stdout.flush()

data = np.load('data/processed/features_aligned.npz')
text_features = data['text_features'].astype(np.float32)

y_dummy = np.zeros(len(text_features))
y_dummy[:100] = 1

print("Building LightGBM Dataset...")
sys.stdout.flush()

try:
    lgb_train = lgb.Dataset(text_features, y_dummy)
    lgb_train.construct()
    print("Construct successful!")
    sys.stdout.flush()
except Exception as e:
    print(f"Failed: {e}")
    sys.stdout.flush()

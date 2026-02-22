import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings

warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

def aggregate_instance(folder_path):
    text_path = folder_path / 'sbert.csv'
    audio_path = folder_path / 'features.csv'
    
    if not text_path.exists() or not audio_path.exists():
        return None
        
    try:
        audio_df = pd.read_csv(audio_path, na_values=['--undefined--'])
        audio_df.columns = audio_df.columns.str.strip()
        for col in audio_df.columns:
            audio_df[col] = pd.to_numeric(audio_df[col], errors='coerce')
        
        audio_data = audio_df.values
        text_df = pd.read_csv(text_path)
        text_data = text_df.values
        
        if len(text_data) != len(audio_data):
            return None
            
        audio_mean = np.nanmean(audio_data, axis=0)
        audio_std = np.nanstd(audio_data, axis=0)
        audio_min = np.nanmin(audio_data, axis=0)
        audio_max = np.nanmax(audio_data, axis=0)
        
        text_mean = np.nanmean(text_data, axis=0)
        text_std = np.nanstd(text_data, axis=0)
        text_min = np.nanmin(text_data, axis=0)
        text_max = np.nanmax(text_data, axis=0)
        
        audio_agg = np.concatenate([audio_mean, audio_std, audio_min, audio_max])
        text_agg = np.concatenate([text_mean, text_std, text_min, text_max])
        
        has_nan = bool(np.isnan(audio_agg).any() or np.isnan(text_agg).any())
        
        return folder_path.name, audio_agg, text_agg, len(text_data), has_nan
        
    except Exception as e:
        print(f"Error processing {folder_path.name}: {e}", file=sys.stderr)
        return None

def main():
    root = Path('data/MAEC_Dataset')
    folders = [f for f in root.iterdir() if f.is_dir()]
    folders.sort()
    
    print(f"Aggregating {len(folders)} calls directly to mmap...")
    sys.stdout.flush()
    Path('data/processed').mkdir(exist_ok=True)
    
    max_len = len(folders)
    instance_ids = []
    audio_features = np.zeros((max_len, 116), dtype=np.float32)
    text_features = np.zeros((max_len, 1536), dtype=np.float32)
    num_sentences = np.zeros(max_len, dtype=np.int32)
    has_nan_arr = np.zeros(max_len, dtype=bool)
    
    valid_idx = 0
    for i, f in enumerate(folders):
        res = aggregate_instance(f)
        if res is not None:
            inst_id, a_agg, t_agg, n_sent, has_nan = res
            
            if valid_idx == 0:
                audio_features = np.zeros((max_len, a_agg.shape[0]), dtype=np.float32)
                text_features = np.zeros((max_len, t_agg.shape[0]), dtype=np.float32)
                
            instance_ids.append(inst_id)
            audio_features[valid_idx] = a_agg
            text_features[valid_idx] = t_agg
            num_sentences[valid_idx] = n_sent
            has_nan_arr[valid_idx] = has_nan
            valid_idx += 1
            
        if (i+1) % 500 == 0:
            print(f"Processed {i+1} / {max_len} folders...")
            sys.stdout.flush()
            
    print(f"Completed loop. Trimming arrays to {valid_idx} valid instances.")
    sys.stdout.flush()

    instance_ids = np.array(instance_ids, dtype=object)
    audio_features = audio_features[:valid_idx]
    text_features = text_features[:valid_idx]
    num_sentences = num_sentences[:valid_idx]
    has_nan_arr = has_nan_arr[:valid_idx]
    
    if has_nan_arr.any():
        num_nan = has_nan_arr.sum()
        print(f"Applying cross-instance mean imputation for {num_nan} anomalous calls...")
        col_means = np.nanmean(audio_features, axis=0)
        inds = np.where(np.isnan(audio_features))
        audio_features[inds] = np.take(col_means, inds[1])
            
    print("Saving to compressed specific format...")
    sys.stdout.flush()
    np.savez_compressed('data/processed/aggregated_features.npz', 
                        instance_ids=instance_ids, 
                        audio_features=audio_features, 
                        text_features=text_features,
                        num_sentences=num_sentences)
    
    print(f"\nSaved {valid_idx} aggregated instances to data/processed/aggregated_features.npz")
    print("Shape of Audio Features:", audio_features.shape)
    print("Shape of Text Features:", text_features.shape)

if __name__ == "__main__":
    main()

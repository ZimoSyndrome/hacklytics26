#!/usr/bin/env python3
"""
Build labels and temporal company-grouped train/val/test split using 7-stat features.

Loads:
  - text_call_matrix.csv  (3443 × 2688 features, 7 temporal stats)
  - audio_call_matrix.csv (3443 × 203 features, 7 temporal stats)
  - final_fraud_data.csv  (SEC AAER fraud incidents)

Produces:
  - data/processed/labels.csv        (instance_id, ticker, call_date, Y_4, Y_8, Y_16, split)
  - data/processed/features.npz      (text_features, audio_features, split)

Split: Temporal + company-grouped (60/20/20).
  All calls for a ticker are assigned to one split.
  Tickers sorted by earliest call date, then assigned by cumulative instance count.

Usage:
    python3 build_labels.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

K_QUARTERS = [4, 8, 16]
DAYS_PER_QUARTER = 91.25


def load_features():
    """Load 7-stat feature matrices from CSVs."""
    print("Loading 7-stat text features from text_call_matrix.csv...")
    text_df = pd.read_csv('data/processed/text_call_matrix.csv')

    print("Loading 7-stat audio features from audio_call_matrix.csv...")
    audio_df = pd.read_csv('data/processed/audio_call_matrix.csv')

    # Extract identifiers (first 3 columns: instance_id, ticker, call_date)
    assert list(text_df.columns[:3]) == ['instance_id', 'ticker', 'call_date']
    assert list(audio_df.columns[:3]) == ['instance_id', 'ticker', 'call_date']

    # Verify row alignment
    assert (text_df['instance_id'].values == audio_df['instance_id'].values).all(), \
        "Instance IDs do not match between text and audio matrices!"

    df_calls = pd.DataFrame({
        'instance_id': text_df['instance_id'].values,
        'ticker': text_df['ticker'].values,
        'call_date': pd.to_datetime(text_df['call_date'].values),
    })

    text_features = text_df.iloc[:, 3:].values.astype(np.float64)
    audio_features = audio_df.iloc[:, 3:].values.astype(np.float64)

    print(f"  Text features shape:  {text_features.shape}")
    print(f"  Audio features shape: {audio_features.shape}")
    print(f"  Instances:            {len(df_calls)}")

    return df_calls, text_features, audio_features


def generate_labels(df_calls):
    """Generate cumulative fraud labels for K-quarter horizons."""
    print("\nLoading fraud data...")
    df_fraud = pd.read_csv('data/final_fraud_data.csv')
    df_fraud['date'] = pd.to_datetime(df_fraud['date'])

    print("Generating cumulative labels (next-fraud-only)...")
    labels_cum = {k: np.zeros(len(df_calls), dtype=np.int8) for k in K_QUARTERS}

    for idx, call in df_calls.iterrows():
        t_call = call['call_date']
        ticker = call['ticker']
        future_frauds = df_fraud[(df_fraud['ticker'] == ticker) & (df_fraud['date'] > t_call)]

        if not future_frauds.empty:
            t_star = future_frauds['date'].min()  # next fraud
            delta_d = (t_star - t_call).days
            q = int(np.ceil(delta_d / DAYS_PER_QUARTER))
            for k in K_QUARTERS:
                if q <= k:
                    labels_cum[k][idx] = 1

    for k in K_QUARTERS:
        df_calls[f'Y_{k}'] = labels_cum[k]
        n_pos = labels_cum[k].sum()
        print(f"  Y_{k}: {n_pos} positives ({n_pos/len(df_calls)*100:.2f}%)")

    return df_calls


def temporal_company_split(df_calls, train_frac=0.60, val_frac=0.20):
    """Temporal + company-grouped split.

    1. For each ticker, find earliest call_date.
    2. Sort tickers by earliest call_date.
    3. Walk sorted tickers, accumulating instances.
    4. First 60% → train, next 20% → val, last 20% → test.
    5. ALL calls for a ticker go to its assigned split.
    """
    print(f"\nPerforming temporal company-grouped split ({train_frac:.0%}/{val_frac:.0%}/{1-train_frac-val_frac:.0%})...")

    # Earliest call per ticker
    ticker_earliest = df_calls.groupby('ticker')['call_date'].min().reset_index()
    ticker_earliest.columns = ['ticker', 'earliest_date']
    ticker_earliest = ticker_earliest.sort_values('earliest_date').reset_index(drop=True)

    # Count instances per ticker
    ticker_counts = df_calls.groupby('ticker').size().to_dict()
    ticker_earliest['n_instances'] = ticker_earliest['ticker'].map(ticker_counts)
    ticker_earliest['cum_instances'] = ticker_earliest['n_instances'].cumsum()

    total = len(df_calls)
    train_cutoff = int(total * train_frac)
    val_cutoff = int(total * (train_frac + val_frac))

    # Assign tickers to splits
    train_tickers = set()
    val_tickers = set()
    test_tickers = set()

    for _, row in ticker_earliest.iterrows():
        if row['cum_instances'] <= train_cutoff:
            train_tickers.add(row['ticker'])
        elif row['cum_instances'] <= val_cutoff:
            val_tickers.add(row['ticker'])
        else:
            test_tickers.add(row['ticker'])

    # Build split array
    split_arr = np.array(['unassigned'] * len(df_calls), dtype=object)
    for idx, row in df_calls.iterrows():
        t = row['ticker']
        if t in train_tickers:
            split_arr[idx] = 'train'
        elif t in val_tickers:
            split_arr[idx] = 'val'
        else:
            split_arr[idx] = 'test'

    df_calls['split'] = split_arr

    # Report split statistics
    for split_name in ['train', 'val', 'test']:
        mask = split_arr == split_name
        n = mask.sum()
        tickers_in_split = df_calls.loc[mask, 'ticker'].nunique()
        date_min = df_calls.loc[mask, 'call_date'].min()
        date_max = df_calls.loc[mask, 'call_date'].max()
        print(f"  {split_name:5s}: {n:4d} instances, {tickers_in_split:4d} tickers, "
              f"dates {date_min.strftime('%Y-%m-%d')} to {date_max.strftime('%Y-%m-%d')}")
        for k in K_QUARTERS:
            n_pos = df_calls.loc[mask, f'Y_{k}'].sum()
            print(f"         Y_{k}: {n_pos} positives")

    # Verify no ticker overlap
    assert len(train_tickers & val_tickers) == 0, "Ticker overlap between train and val!"
    assert len(train_tickers & test_tickers) == 0, "Ticker overlap between train and test!"
    assert len(val_tickers & test_tickers) == 0, "Ticker overlap between val and test!"
    print("\n  ✓ No ticker overlap between splits")

    # Check fraud distribution — warn if any split has 0 fraud at K=16
    for split_name in ['train', 'val', 'test']:
        mask = split_arr == split_name
        n_fraud = df_calls.loc[mask, 'Y_16'].sum()
        if n_fraud == 0:
            print(f"\n  ⚠ WARNING: {split_name} has 0 fraud instances at K=16!")
            print("  Consider adjusting split boundaries.")

    return df_calls


def save_outputs(df_calls, text_features, audio_features):
    """Save labels CSV and features NPZ."""
    out_dir = Path('data/processed')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save labels
    label_cols = ['instance_id', 'ticker', 'call_date'] + [f'Y_{k}' for k in K_QUARTERS] + ['split']
    df_out = df_calls[label_cols].copy()
    df_out['call_date'] = df_out['call_date'].dt.strftime('%Y-%m-%d')
    df_out.to_csv(out_dir / 'labels.csv', index=False)
    print(f"\nSaved labels to {out_dir / 'labels.csv'}")

    # Save features
    np.savez_compressed(
        out_dir / 'features.npz',
        text_features=text_features,
        audio_features=audio_features,
        split=df_calls['split'].values,
    )
    print(f"Saved features to {out_dir / 'features.npz'}")
    print(f"  text_features:  {text_features.shape}")
    print(f"  audio_features: {audio_features.shape}")


def main():
    df_calls, text_features, audio_features = load_features()
    df_calls = generate_labels(df_calls)
    df_calls = temporal_company_split(df_calls)
    save_outputs(df_calls, text_features, audio_features)

    print("\n" + "=" * 60)
    print("BUILD LABELS COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

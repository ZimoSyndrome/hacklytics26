#!/usr/bin/env python3
"""
Train LightGBM fraud classifiers using 7-stat features with late fusion.

Loads:
  - data/processed/labels.csv       (temporal company-grouped split)
  - data/processed/features.npz     (7-stat text 2688d + audio 203d)

Models trained per horizon (K=4, 8, 16):
  1. Text-only   (2688 dims)
  2. Audio-only   (203 dims)
  3. Late fusion   (weighted average of text + audio probabilities, weight from val)

Evaluation:
  - Isotonic calibration on val
  - F1-optimal threshold on val
  - Cluster bootstrap PR-AUC CI on test (300 iterations, ticker-level)
  - Feature importance (top 20 by gain)

Usage:
    python3 train_models.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, average_precision_score, brier_score_loss,
)
from sklearn.isotonic import IsotonicRegression
from sklearn.utils import resample
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

K_QUARTERS = [4, 8, 16]


def load_data():
    """Load features and labels."""
    df = pd.read_csv('data/processed/labels.csv')
    data = np.load('data/processed/features.npz', allow_pickle=True)

    text = data['text_features']
    audio = data['audio_features']

    X = {'text': text, 'audio': audio}
    return df, X


def cluster_bootstrap_pr_auc(y_true, y_prob, tickers, n_iterations=300):
    """Bootstrap PR-AUC with ticker-level resampling."""
    unique_tickers = np.unique(tickers)
    pr_aucs = []

    if len(np.unique(y_true)) < 2:
        return 0.0, 0.0, 0.0

    df_temp = pd.DataFrame({'y': y_true, 'prob': y_prob, 'ticker': tickers})

    for _ in range(n_iterations):
        boot_tickers = resample(unique_tickers, n_samples=len(unique_tickers))
        boot_rows = []
        for t in boot_tickers:
            boot_rows.append(df_temp[df_temp['ticker'] == t])

        boot_df = pd.concat(boot_rows)
        y_boot = boot_df['y'].values
        prob_boot = boot_df['prob'].values

        if len(np.unique(y_boot)) > 1:
            pr_aucs.append(average_precision_score(y_boot, prob_boot))

    if len(pr_aucs) == 0:
        return 0.0, 0.0, 0.0

    pr_aucs = np.array(pr_aucs)
    return np.mean(pr_aucs), np.percentile(pr_aucs, 2.5), np.percentile(pr_aucs, 97.5)


def evaluate_model(y_test, preds, threshold=0.5):
    """Compute all evaluation metrics."""
    preds_bin = (preds >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds_bin)

    acc = accuracy_score(y_test, preds_bin)
    prec = precision_score(y_test, preds_bin, zero_division=0)
    rec = recall_score(y_test, preds_bin, zero_division=0)
    f1 = f1_score(y_test, preds_bin, zero_division=0)
    pr_auc = average_precision_score(y_test, preds)
    brier = brier_score_loss(y_test, preds)

    return cm, acc, prec, rec, f1, pr_auc, brier


def find_best_threshold(y_val, val_preds, fallback_prior):
    """Find F1-maximizing threshold on validation set."""
    best_f1 = 0
    best_thresh = 0.5
    for th in np.linspace(0.01, 0.5, 50):
        preds_bin = (val_preds >= th).astype(int)
        f = f1_score(y_val, preds_bin, zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_thresh = th

    if best_thresh < 0.01 or best_f1 == 0:
        best_thresh = fallback_prior

    return best_thresh, best_f1


def train_single_modality(X_tr, y_train, X_v, y_val, base_params):
    """Train a single LightGBM model, return model + calibrated val/test predictor."""
    pos_ratio = sum(y_train == 0) / max(1, sum(y_train == 1))
    current_params = base_params.copy()
    current_params['scale_pos_weight'] = min(50.0, pos_ratio / 2)

    lgb_train = lgb.Dataset(X_tr, y_train, free_raw_data=False).construct()
    lgb_val = lgb.Dataset(X_v, y_val, reference=lgb_train, free_raw_data=False).construct()

    model = lgb.train(
        current_params,
        lgb_train,
        num_boost_round=300,
        valid_sets=[lgb_val],
        callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
    )

    # Calibrate on validation
    val_preds_raw = model.predict(X_v, num_iteration=model.best_iteration)
    calibrator = IsotonicRegression(out_of_bounds='clip')
    calibrator.fit(val_preds_raw, y_val)

    val_preds_cal = calibrator.predict(val_preds_raw)

    return model, calibrator, val_preds_cal


def get_feature_importance(model, feature_names, top_n=20):
    """Get top N features by gain importance."""
    importance = model.feature_importance(importance_type='gain')
    idx_sorted = np.argsort(importance)[::-1][:top_n]
    return [(feature_names[i], float(importance[i])) for i in idx_sorted if importance[i] > 0]


def print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, threshold):
    """Print formatted evaluation results."""
    print(f"\nConfusion Matrix (Optimal F1 Threshold = {threshold:.3f}):")
    if cm.shape == (2, 2):
        print(f"[{cm[0][0]:4d}  {cm[0][1]:4d}]  (TN  FP)")
        print(f"[{cm[1][0]:4d}  {cm[1][1]:4d}]  (FN  TP)")
    elif cm.shape == (1, 1):
        print(f"[{cm[0][0]:4d}     0]  (TN  FP)")
        print(f"[   0     0]  (FN  TP)")
    else:
        print(cm)

    print(f"\nMetrics:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Brier:     {brier:.4f}")
    print(f"  PR-AUC:    {pr_auc:.4f} (95% CI: {pr_lower:.4f} - {pr_upper:.4f})")


def main():
    df, X = load_data()
    splits = df['split'].values
    train_idx = np.where(splits == 'train')[0]
    val_idx = np.where(splits == 'val')[0]
    test_idx = np.where(splits == 'test')[0]

    base_params = {
        'objective': 'binary',
        'metric': 'average_precision',
        'boosting_type': 'gbdt',
        'learning_rate': 0.05,
        'num_leaves': 15,
        'max_depth': 4,
        'feature_fraction': 0.8,
        'verbose': -1,
        'n_jobs': 1,
    }

    # Load feature names for importance analysis
    text_df = pd.read_csv('data/processed/text_call_matrix.csv', nrows=0)
    audio_df = pd.read_csv('data/processed/audio_call_matrix.csv', nrows=0)
    text_feature_names = list(text_df.columns[3:])
    audio_feature_names = list(audio_df.columns[3:])

    results_log = {
        'started_at': datetime.now().isoformat(),
        'features': '7-stat (mean, std, max, min, first, last, slope)',
        'split': 'temporal_company_grouped',
        'text_dims': len(text_feature_names),
        'audio_dims': len(audio_feature_names),
        'results': [],
    }

    print("=" * 70)
    print("LIGHTGBM V2 — 7-STAT FEATURES + TEMPORAL COMPANY-GROUPED SPLIT")
    print("=" * 70)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    for k in K_QUARTERS:
        target_col = f'Y_{k}'
        y = df[target_col].values

        y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

        if sum(y_train) == 0:
            print(f"\nSkipping K={k}: 0 positives in train.")
            continue
        if sum(y_test) == 0:
            print(f"\nSkipping K={k}: 0 positives in test.")
            continue

        # ---- TEXT-ONLY ----
        print(f"\n{'='*60}")
        print(f"MODALITY: TEXT  |  HORIZON: K={k} Quarters")
        print(f"{'='*60}")
        print(f"Train: {sum(y_train==1)} Pos/{sum(y_train==0)} Neg | "
              f"Val: {sum(y_val==1)} Pos/{sum(y_val==0)} Neg | "
              f"Test: {sum(y_test==1)} Pos/{sum(y_test==0)} Neg")

        X_text_tr = X['text'][train_idx]
        X_text_v = X['text'][val_idx]
        X_text_tt = X['text'][test_idx]

        text_model, text_cal, text_val_preds = train_single_modality(
            X_text_tr, y_train, X_text_v, y_val, base_params
        )

        fallback = sum(y_val == 1) / len(y_val)
        text_thresh, _ = find_best_threshold(y_val, text_val_preds, fallback)

        text_test_raw = text_model.predict(X_text_tt, num_iteration=text_model.best_iteration)
        text_test_preds = text_cal.predict(text_test_raw)

        test_tickers = df['ticker'].values[test_idx]
        cm, acc, prec, rec, f1, pr_auc, brier = evaluate_model(y_test, text_test_preds, text_thresh)
        pr_mean, pr_lower, pr_upper = cluster_bootstrap_pr_auc(y_test, text_test_preds, test_tickers)
        print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, text_thresh)

        # Feature importance
        top_text_feats = get_feature_importance(text_model, text_feature_names, top_n=20)
        if top_text_feats:
            print(f"\nTop text features (by gain):")
            for fname, fval in top_text_feats[:10]:
                print(f"  {fname}: {fval:.1f}")

        results_log['results'].append({
            'horizon': k, 'modality': 'text', 'pr_auc': pr_auc,
            'pr_ci': [pr_lower, pr_upper], 'f1': f1, 'threshold': text_thresh,
            'top_features': top_text_feats[:10],
        })

        # ---- AUDIO-ONLY ----
        print(f"\n{'='*60}")
        print(f"MODALITY: AUDIO  |  HORIZON: K={k} Quarters")
        print(f"{'='*60}")
        print(f"Train: {sum(y_train==1)} Pos/{sum(y_train==0)} Neg | "
              f"Val: {sum(y_val==1)} Pos/{sum(y_val==0)} Neg | "
              f"Test: {sum(y_test==1)} Pos/{sum(y_test==0)} Neg")

        X_audio_tr = X['audio'][train_idx]
        X_audio_v = X['audio'][val_idx]
        X_audio_tt = X['audio'][test_idx]

        audio_model, audio_cal, audio_val_preds = train_single_modality(
            X_audio_tr, y_train, X_audio_v, y_val, base_params
        )

        audio_thresh, _ = find_best_threshold(y_val, audio_val_preds, fallback)

        audio_test_raw = audio_model.predict(X_audio_tt, num_iteration=audio_model.best_iteration)
        audio_test_preds = audio_cal.predict(audio_test_raw)

        cm, acc, prec, rec, f1, pr_auc, brier = evaluate_model(y_test, audio_test_preds, audio_thresh)
        pr_mean, pr_lower, pr_upper = cluster_bootstrap_pr_auc(y_test, audio_test_preds, test_tickers)
        print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, audio_thresh)

        # Feature importance
        top_audio_feats = get_feature_importance(audio_model, audio_feature_names, top_n=20)
        if top_audio_feats:
            print(f"\nTop audio features (by gain):")
            for fname, fval in top_audio_feats[:10]:
                print(f"  {fname}: {fval:.1f}")

        results_log['results'].append({
            'horizon': k, 'modality': 'audio', 'pr_auc': pr_auc,
            'pr_ci': [pr_lower, pr_upper], 'f1': f1, 'threshold': audio_thresh,
            'top_features': top_audio_feats[:10],
        })

        # ---- LATE FUSION ----
        print(f"\n{'='*60}")
        print(f"MODALITY: LATE FUSION  |  HORIZON: K={k} Quarters")
        print(f"{'='*60}")

        # Find optimal fusion weight on val set
        best_w, best_ap = 0.5, 0.0
        for w in np.arange(0.0, 1.01, 0.05):
            p_fused = w * text_val_preds + (1 - w) * audio_val_preds
            if len(np.unique(y_val)) > 1:
                ap = average_precision_score(y_val, p_fused)
                if ap > best_ap:
                    best_w, best_ap = w, ap

        print(f"Optimal fusion weight: {best_w:.2f} text + {1-best_w:.2f} audio (val AP={best_ap:.4f})")

        # Apply fusion to test
        fused_test_preds = best_w * text_test_preds + (1 - best_w) * audio_test_preds

        fused_thresh, _ = find_best_threshold(
            y_val,
            best_w * text_val_preds + (1 - best_w) * audio_val_preds,
            fallback
        )

        cm, acc, prec, rec, f1, pr_auc, brier = evaluate_model(y_test, fused_test_preds, fused_thresh)
        pr_mean, pr_lower, pr_upper = cluster_bootstrap_pr_auc(y_test, fused_test_preds, test_tickers)
        print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, fused_thresh)

        results_log['results'].append({
            'horizon': k, 'modality': 'late_fusion', 'pr_auc': pr_auc,
            'pr_ci': [pr_lower, pr_upper], 'f1': f1, 'threshold': fused_thresh,
            'fusion_weight_text': best_w,
        })

    # Save results log
    results_dir = Path('data/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    results_log['completed_at'] = datetime.now().isoformat()

    with open(results_dir / 'model_results.json', 'w') as f:
        json.dump(results_log, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("TRAINING COMPLETE — Results saved to data/results/model_results.json")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()

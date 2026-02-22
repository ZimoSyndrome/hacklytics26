#!/usr/bin/env python3
"""
Train LightGBM fraud classifiers using 7-stat features with late fusion.

Loads:
  - data/processed/labels.csv              (temporal company-grouped split)
  - data/processed/features.npz            (7-stat text 2688d + audio 203d)
  - data/processed/filings_call_matrix.csv (10-K/10-Q features 790d) [optional]

Models trained per horizon (K=4, 8, 16):
  1. Text-only     (2688 dims)
  2. Audio-only     (203 dims)
  3. Late fusion     (weighted average of text + audio, weight from val)
  4. Filings-only   (790 dims) — if filings_call_matrix.csv exists
  5. 3-way fusion   (text + audio + filings, 2D grid search on val) — if filings exist

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


def load_filings_features(
    path: Path,
    instance_order: list,
) -> tuple:
    """Load filings_call_matrix.csv and return the feature matrix + coverage mask.

    Verifies that the instance_id column matches instance_order exactly.

    Returns:
        X_filings: np.ndarray shape (N, 790) — feature columns (has_10k, has_10q, embeddings, lexical)
        has_filing: np.ndarray shape (N,) bool — True if instance has at least one filing
    """
    df_fil = pd.read_csv(path)
    fil_ids = df_fil['instance_id'].tolist()

    if fil_ids != instance_order:
        mismatches = [(i, a, b) for i, (a, b) in enumerate(zip(fil_ids, instance_order)) if a != b]
        raise ValueError(
            f"filings_call_matrix.csv row order does not match labels.csv. "
            f"First mismatch at index {mismatches[0][0]}: '{mismatches[0][1]}' vs '{mismatches[0][2]}'"
        )

    # Coverage mask: has at least one filing
    has_filing = ((df_fil['has_10k'] == 1) | (df_fil['has_10q'] == 1)).values

    # Feature columns start after instance_id, ticker, call_date
    feat_cols = [c for c in df_fil.columns if c not in ('instance_id', 'ticker', 'call_date')]
    X_filings = df_fil[feat_cols].values.astype(np.float64)
    print(f"  Filings features shape: {X_filings.shape}")
    n_covered = has_filing.sum()
    print(f"  Instances with >=1 filing: {n_covered}/{len(has_filing)} ({n_covered/len(has_filing)*100:.1f}%)")
    return X_filings, has_filing


def grid_search_3way_fusion(
    p_text: np.ndarray,
    p_audio: np.ndarray,
    p_filings: np.ndarray,
    y_val: np.ndarray,
    has_filing_val: np.ndarray,
    n_steps: int = 11,
) -> tuple:
    """Grid search over (w_text, w_audio, w_filings) where weights sum to 1.

    Grid step = 1/(n_steps-1) = 0.10 with n_steps=11.
    Generates all valid (w_t, w_a, w_f) triplets with w_f = 1 - w_t - w_a >= 0.

    Weight search is performed ONLY on instances that have at least one filing,
    so the filing signal isn't diluted by zero-padded rows (Option B: conditional fusion).
    Scores each by Average Precision on the filing-covered validation subset.

    Returns:
        (best_w_text, best_w_audio, best_w_filings, best_val_ap)
    """
    # Use only filing-covered val instances to find weights
    covered_mask = has_filing_val
    if covered_mask.sum() > 0 and len(np.unique(y_val[covered_mask])) > 1:
        p_t_c = p_text[covered_mask]
        p_a_c = p_audio[covered_mask]
        p_f_c = p_filings[covered_mask]
        y_c   = y_val[covered_mask]
    else:
        # Fallback: use all val instances
        p_t_c, p_a_c, p_f_c, y_c = p_text, p_audio, p_filings, y_val

    if len(np.unique(y_c)) < 2:
        return 1 / 3, 1 / 3, 1 / 3, 0.0

    step = 1.0 / (n_steps - 1)
    best_ap = -1.0
    best_weights = (1 / 3, 1 / 3, 1 / 3)

    for i in range(n_steps):
        w_t = round(i * step, 8)
        for j in range(n_steps - i):
            w_a = round(j * step, 8)
            w_f = round(1.0 - w_t - w_a, 8)
            if w_f < -1e-9:
                continue
            w_f = max(0.0, w_f)
            p_fused = w_t * p_t_c + w_a * p_a_c + w_f * p_f_c
            ap = average_precision_score(y_c, p_fused)
            if ap > best_ap:
                best_ap = ap
                best_weights = (w_t, w_a, w_f)

    return (*best_weights, best_ap)


def apply_conditional_fusion(
    p_text: np.ndarray,
    p_audio: np.ndarray,
    p_filings: np.ndarray,
    has_filing: np.ndarray,
    w_t: float,
    w_a: float,
    w_f: float,
) -> np.ndarray:
    """Apply 3-way fusion with conditional fallback for missing-filing instances.

    For instances WITH filings: p = w_t*p_text + w_a*p_audio + w_f*p_filings
    For instances WITHOUT filings: p = (w_t/(w_t+w_a))*p_text + (w_a/(w_t+w_a))*p_audio
    This avoids injecting noise from the filings model's zero-feature predictions
    into the ensemble score for instances that have no filing data.
    """
    p_fused = np.empty(len(p_text))
    # Instances with filings: full 3-way blend
    mask = has_filing
    p_fused[mask] = w_t * p_text[mask] + w_a * p_audio[mask] + w_f * p_filings[mask]
    # Instances without filings: renormalised 2-way blend
    no_mask = ~mask
    denom = w_t + w_a if (w_t + w_a) > 0 else 1.0
    p_fused[no_mask] = (w_t / denom) * p_text[no_mask] + (w_a / denom) * p_audio[no_mask]
    return p_fused


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

    # Optionally load filings features if the matrix exists
    filings_path = Path('data/processed/filings_call_matrix.csv')
    X_filings = None
    has_filing_mask = None   # bool array (N,): True if instance has >=1 filing
    filings_feature_names: list = []
    if filings_path.exists():
        print(f"[INFO] Loading filings features from {filings_path}")
        instance_order = df['instance_id'].tolist()
        X_filings, has_filing_mask = load_filings_features(filings_path, instance_order)
        # Column names: skip instance_id, ticker, call_date
        fil_df_header = pd.read_csv(filings_path, nrows=0)
        filings_feature_names = [
            c for c in fil_df_header.columns
            if c not in ('instance_id', 'ticker', 'call_date')
        ]
        X['filings'] = X_filings
    else:
        print(f"[INFO] {filings_path} not found — skipping filings modality.")

    results_log = {
        'started_at': datetime.now().isoformat(),
        'features': '7-stat (mean, std, max, min, first, last, slope)',
        'split': 'temporal_company_grouped',
        'text_dims': len(text_feature_names),
        'audio_dims': len(audio_feature_names),
        'filings_dims': len(filings_feature_names) if filings_feature_names else 0,
        'results': [],
    }

    print("=" * 70)
    print("LIGHTGBM — 7-STAT FEATURES + TEMPORAL COMPANY-GROUPED SPLIT")
    print("=" * 70)
    print(f"Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)}")

    # Store per-K calibrated test predictions for 3-way fusion
    stored_preds: dict = {}  # k -> {'text_val', 'audio_val', 'text_test', 'audio_test'}

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

        # Store calibrated predictions for this horizon (used later for 3-way fusion)
        stored_preds[k] = {
            'text_val': text_val_preds,
            'audio_val': audio_val_preds,
            'text_test': text_test_preds,
            'audio_test': audio_test_preds,
        }

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

    # =========================================================
    # FILINGS MODALITY + 3-WAY FUSION (if matrix available)
    # =========================================================
    if X_filings is not None:
        print(f"\n{'='*70}")
        print("FILINGS MODALITY + 3-WAY LATE FUSION")
        print(f"{'='*70}")
        print(f"Filings dims: {X_filings.shape[1]}")

        for k in K_QUARTERS:
            target_col = f'Y_{k}'
            y = df[target_col].values
            y_train, y_val, y_test = y[train_idx], y[val_idx], y[test_idx]

            if sum(y_train) == 0 or sum(y_test) == 0:
                print(f"\nSkipping K={k} filings/3-way: insufficient positives.")
                continue

            if k not in stored_preds:
                print(f"\nSkipping K={k} 3-way fusion: no stored text/audio preds.")
                continue

            fallback = sum(y_val == 1) / len(y_val)
            test_tickers = df['ticker'].values[test_idx]

            # ---- FILINGS-ONLY ----
            print(f"\n{'='*60}")
            print(f"MODALITY: FILINGS  |  HORIZON: K={k} Quarters")
            print(f"{'='*60}")
            print(f"Train: {sum(y_train==1)} Pos/{sum(y_train==0)} Neg | "
                  f"Val: {sum(y_val==1)} Pos/{sum(y_val==0)} Neg | "
                  f"Test: {sum(y_test==1)} Pos/{sum(y_test==0)} Neg")

            X_fil_tr = X_filings[train_idx]
            X_fil_v = X_filings[val_idx]
            X_fil_tt = X_filings[test_idx]

            fil_model, fil_cal, fil_val_preds = train_single_modality(
                X_fil_tr, y_train, X_fil_v, y_val, base_params
            )

            fil_thresh, _ = find_best_threshold(y_val, fil_val_preds, fallback)

            fil_test_raw = fil_model.predict(X_fil_tt, num_iteration=fil_model.best_iteration)
            fil_test_preds = fil_cal.predict(fil_test_raw)

            cm, acc, prec, rec, f1, pr_auc, brier = evaluate_model(y_test, fil_test_preds, fil_thresh)
            pr_mean, pr_lower, pr_upper = cluster_bootstrap_pr_auc(y_test, fil_test_preds, test_tickers)
            print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, fil_thresh)

            top_fil_feats = get_feature_importance(fil_model, filings_feature_names, top_n=20)
            if top_fil_feats:
                print(f"\nTop filings features (by gain):")
                for fname, fval in top_fil_feats[:10]:
                    print(f"  {fname}: {fval:.1f}")

            n_test_with_filing = has_filing_mask[test_idx].sum()
            results_log['results'].append({
                'horizon': k, 'modality': 'filings', 'pr_auc': pr_auc,
                'pr_ci': [pr_lower, pr_upper], 'f1': f1, 'threshold': fil_thresh,
                'top_features': top_fil_feats[:10],
                'test_coverage_pct': round(n_test_with_filing / len(test_idx) * 100, 1),
            })

            # ---- 3-WAY LATE FUSION (conditional) ----
            print(f"\n{'='*60}")
            print(f"MODALITY: 3-WAY FUSION (text+audio+filings)  |  HORIZON: K={k} Quarters")
            print(f"{'='*60}")

            p_text_val = stored_preds[k]['text_val']
            p_audio_val = stored_preds[k]['audio_val']
            p_text_test = stored_preds[k]['text_test']
            p_audio_test = stored_preds[k]['audio_test']

            # Coverage masks for val/test splits
            has_filing_val_mask = has_filing_mask[val_idx]
            has_filing_test_mask = has_filing_mask[test_idx]
            n_val_covered = has_filing_val_mask.sum()
            n_test_covered = has_filing_test_mask.sum()
            print(f"Val coverage: {n_val_covered}/{len(val_idx)} have filings | "
                  f"Test coverage: {n_test_covered}/{len(test_idx)} have filings")

            # Weight search restricted to filing-covered val instances
            w_t, w_a, w_f, val_ap = grid_search_3way_fusion(
                p_text_val, p_audio_val, fil_val_preds, y_val, has_filing_val_mask
            )
            print(f"Optimal weights (on covered val): {w_t:.2f} text + {w_a:.2f} audio + "
                  f"{w_f:.2f} filings (val AP={val_ap:.4f})")

            # Conditional fusion: renormalise to 2-way for instances without filings
            fused3_test = apply_conditional_fusion(
                p_text_test, p_audio_test, fil_test_preds,
                has_filing_test_mask, w_t, w_a, w_f
            )

            # Threshold search on full val (conditional fusion applied consistently)
            fused3_val = apply_conditional_fusion(
                p_text_val, p_audio_val, fil_val_preds,
                has_filing_val_mask, w_t, w_a, w_f
            )
            fused3_thresh, _ = find_best_threshold(y_val, fused3_val, fallback)

            cm, acc, prec, rec, f1, pr_auc, brier = evaluate_model(y_test, fused3_test, fused3_thresh)
            pr_mean, pr_lower, pr_upper = cluster_bootstrap_pr_auc(y_test, fused3_test, test_tickers)
            print_results(cm, acc, prec, rec, f1, pr_auc, brier, pr_lower, pr_upper, fused3_thresh)

            results_log['results'].append({
                'horizon': k,
                'modality': 'text_audio_filings_fusion',
                'pr_auc': pr_auc,
                'pr_ci': [pr_lower, pr_upper],
                'f1': f1,
                'threshold': fused3_thresh,
                'fusion_weights': {'text': w_t, 'audio': w_a, 'filings': w_f},
                'fusion_type': 'conditional',
                'test_coverage_pct': round(n_test_covered / len(test_idx) * 100, 1),
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

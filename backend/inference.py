"""
Inference module for the fraud signal analysis backend.

At startup loads:
  - LightGBM models + isotonic calibrators  from backend/config/models/*.joblib
  - Fusion weights, thresholds, top features, PR-AUC from data/results/model_results.json
    (produced by train_models.py)

For each request, extracts features from three inputs:
  Text   (.txt, one sentence per line)
      -> SBERT (all-MiniLM-L6-v2) -> 7-stat temporal compression -> 2688-dim
  Audio  (.csv, 29 acoustic columns, one row per sentence)
      -> parse -> impute NaN -> 7-stat compression -> 203-dim
  Report (.pdf quarterly filing)
      -> pdfplumber text extraction -> SBERT -> 7-stat compression -> 2688-dim
      -> element-wise averaged with transcript text vector

Runs LightGBM models for K = 4, 8, 16 quarter horizons.
Applies isotonic calibration and late fusion with weights from model_results.json.
"""

import io
import json
import pathlib
import csv
import logging
import pandas as pd

import numpy as np
from sentence_transformers import SentenceTransformer

from models import (
    AnalysisResult,
    DistributionPoint,
    FeatureImportance,
    ModelInfo,
    make_feature_label,
    score_to_risk_level,
)

logger = logging.getLogger(__name__)


_SBERT_MODEL = SentenceTransformer("all-MiniLM-L6-v2")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BACKEND_DIR  = pathlib.Path(__file__).parent
_MODELS_DIR   = _BACKEND_DIR / "config" / "models"
_RESULTS_PATH = pathlib.Path(__file__).parent.parent / "data" / "results" / "model_results.json"

K_HORIZONS = [4, 8, 16]   # quarters ahead
_PRIMARY_K  = 8            # horizon used for topFeatures and overallRiskScore

# ---------------------------------------------------------------------------
# Load models and results at startup
# ---------------------------------------------------------------------------

def _try_load(path):
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        logger.debug("Could not load %s: %s", path, e)
        return None


def _load_models() -> dict:
    """Load LightGBM models + isotonic calibrators for each horizon."""
    models = {}
    for k in K_HORIZONS:
        models[k] = {
            "text_model":  _try_load(_MODELS_DIR / f"text_model_k{k}.joblib"),
            "text_cal":    _try_load(_MODELS_DIR / f"text_cal_k{k}.joblib"),
            "audio_model": _try_load(_MODELS_DIR / f"audio_model_k{k}.joblib"),
            "audio_cal":   _try_load(_MODELS_DIR / f"audio_cal_k{k}.joblib"),
        }
    return models


def _load_results() -> dict:
    """Load model_results.json produced by train_models.py."""
    if _RESULTS_PATH.exists():
        return json.loads(_RESULTS_PATH.read_text())
    logger.warning("model_results.json not found at %s", _RESULTS_PATH)
    return {}


_MODELS  = _load_models()
_RESULTS = _load_results()

_MODELS_READY = any(
    slot["text_model"] is not None or slot["audio_model"] is not None
    for slot in _MODELS.values()
)

# ---------------------------------------------------------------------------
# model_results.json helpers
# ---------------------------------------------------------------------------

def _get_entry(k: int, modality: str) -> dict:
    for entry in _RESULTS.get("results", []):
        if entry.get("horizon") == k and entry.get("modality") == modality:
            return entry
    return {}


def _fusion_weight_text(k: int) -> float:
    """Text weight for 2-way late fusion at horizon k."""
    return _get_entry(k, "late_fusion").get("fusion_weight_text", 0.5)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _compute_temporal_stats(matrix: np.ndarray) -> np.ndarray:
    """7-stat temporal compression: [mean, std, max, min, first, last, slope]."""
    T, D = matrix.shape
    if T == 0:
        return np.zeros(D * 7, dtype=np.float64)
    if T == 1:
        val = np.where(np.isnan(matrix[0]), 0.0, matrix[0])
        return np.concatenate([val, np.zeros(D), val, val, val, val, np.zeros(D)])

    def _safe(arr):
        arr[np.isnan(arr)] = 0.0
        return arr

    stat_mean  = _safe(np.nanmean(matrix, axis=0))
    stat_std   = _safe(np.nanstd(matrix, axis=0, ddof=0))
    stat_max   = _safe(np.nanmax(matrix, axis=0))
    stat_min   = _safe(np.nanmin(matrix, axis=0))
    stat_first = np.where(np.isnan(matrix[0]),  0.0, matrix[0])
    stat_last  = np.where(np.isnan(matrix[-1]), 0.0, matrix[-1])

    t = np.linspace(0.0, 1.0, T)
    t_c = t - t.mean()
    var_t = np.dot(t_c, t_c)

    mfs = matrix.copy()
    nan_mask = np.isnan(mfs)
    if nan_mask.any():
        col_means = np.where(np.isnan(np.nanmean(mfs, axis=0)), 0.0,
                             np.nanmean(mfs, axis=0))
        inds = np.where(nan_mask)
        mfs[inds] = np.take(col_means, inds[1])

    mc = mfs - mfs.mean(axis=0, keepdims=True)
    stat_slope = (t_c @ mc) / var_t

    return np.concatenate([stat_mean, stat_std, stat_max, stat_min,
                            stat_first, stat_last, stat_slope])


def _sbert_encode(sentences):
    return np.array(
        _SBERT_MODEL.encode(
            sentences,
            batch_size=128,
            show_progress_bar=False,
            normalize_embeddings=False
        ),
        dtype=np.float64,
    )

def _extract_text_features(text_bytes: bytes) -> np.ndarray | None:
    """Decode transcript, encode sentences with SBERT, apply 7-stat compression."""
    text = text_bytes.decode("utf-8", errors="replace")
    sentences = [s.strip() for s in text.splitlines() if s.strip()]
    if not sentences:
        return None
    try:
        return _compute_temporal_stats(_sbert_encode(sentences))
    except ImportError:
        logger.warning("sentence-transformers not installed; skipping text features")
        return None


def _extract_report_features(report_bytes: bytes) -> np.ndarray | None:
    """Extract text from PDF, encode with SBERT, apply 7-stat compression."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(report_bytes)) as pdf:
            raw_text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        sentences = [s.strip() for s in raw_text.splitlines() if s.strip()]
        if not sentences:
            return None
        return _compute_temporal_stats(_sbert_encode(sentences))
    except ImportError as e:
        logger.warning("Missing dependency for report extraction: %s", e)
        return None
    except Exception as e:
        logger.warning("Report feature extraction failed: %s", e)
        return None


def _extract_audio_features(audio_bytes: bytes):
    try:
        df = pd.read_csv(
            io.BytesIO(audio_bytes),
            engine="python",
            on_bad_lines="skip"
        )

        numeric_df = df.select_dtypes(include=np.number)

        if numeric_df.empty:
            return None

        matrix = numeric_df.values
        matrix = np.nan_to_num(matrix)

        return _compute_temporal_stats(matrix)

    except Exception as e:
        logger.warning(f"Audio feature extraction failed: {e}")
        return None
    
# ---------------------------------------------------------------------------
# Per-horizon LightGBM inference
# ---------------------------------------------------------------------------

def _predict_horizon(k: int, text_vec: np.ndarray | None,
                     audio_vec: np.ndarray | None) -> float:
    """
    Run calibrated LightGBM inference for a single horizon k.
    Applies late-fusion weights from model_results.json.
    Returns a calibrated fraud probability in [0, 1].
    """
    slot = _MODELS[k]
    text_prob  = None
    audio_prob = None

    if slot["text_model"] and slot["text_cal"] and text_vec is not None:
        raw = slot["text_model"].predict(
            text_vec.reshape(1, -1),
            num_iteration=slot["text_model"].best_iteration,
        )
        text_prob = float(slot["text_cal"].predict(raw)[0])

    if slot["audio_model"] and slot["audio_cal"] and audio_vec is not None:
        raw = slot["audio_model"].predict(
            audio_vec.reshape(1, -1),
            num_iteration=slot["audio_model"].best_iteration,
        )
        audio_prob = float(slot["audio_cal"].predict(raw)[0])

    if text_prob is not None and audio_prob is not None:
        w = _fusion_weight_text(k)
        return w * text_prob + (1 - w) * audio_prob
    if text_prob is not None:
        return text_prob
    if audio_prob is not None:
        return audio_prob

    raise RuntimeError(
        f"No models loaded for horizon K={k}. Run train_models.py to generate model files."
    )


# ---------------------------------------------------------------------------
# Result metadata from model_results.json
# ---------------------------------------------------------------------------

def _get_top_features(k: int, modality: str) -> list[FeatureImportance]:
    raw = _get_entry(k, modality).get("top_features", [])
    if not raw:
        return []
    max_imp = max(v for _, v in raw) or 1.0
    return [
        FeatureImportance(
            name=name,
            importance=round(imp / max_imp, 4),
            label=make_feature_label(name),
        )
        for name, imp in raw
    ]


def _get_model_info(k: int, modality: str) -> ModelInfo | None:
    entry = _get_entry(k, modality)
    if not entry:
        return None
    ci = entry.get("pr_ci", [0.0, 0.0])
    return ModelInfo(
        horizon=k,
        modality=modality,
        pr_auc=round(entry.get("pr_auc", 0.0), 4),
        pr_ci_lower=round(ci[0], 4),
        pr_ci_upper=round(ci[1], 4),
        brier=round(entry.get("brier", 0.0) if "brier" in entry else 0.0, 4),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_inference(text_bytes: bytes | None, audio_bytes: bytes | None,
                  report_bytes: bytes | None = None) -> dict:
    """
    Run fraud signal inference on uploaded earnings call inputs.

    Requires trained model files in backend/config/models/ (from train_models.py)
    and data/results/model_results.json for fusion weights and result metadata.

    Returns an AnalysisResult dict:
      overallRiskScore  int 0-100
      riskLevel         "critical" | "high" | "elevated" | "moderate" | "low"
      distribution      [{quarter, probability}]  for K = 4, 8, 16
      topFeatures       [{name, importance, label}]
      modelInfo         {horizon, modality, pr_auc, pr_ci_lower, pr_ci_upper, brier}
    """
    if not _RESULTS:
        raise RuntimeError(
            "model_results.json not found. Run train_models.py first."
        )
    if not _MODELS_READY:
        raise RuntimeError(
            "No trained model files found in backend/config/models/. "
            "Run train_models.py first."
        )

    # --- Feature extraction ---
    text_vec   = _extract_text_features(text_bytes)    if text_bytes   else None
    audio_vec  = _extract_audio_features(audio_bytes)  if audio_bytes  else None
    report_vec = _extract_report_features(report_bytes) if report_bytes else None

    print("text_vec norm:", np.linalg.norm(text_vec) if text_vec is not None else None)
    print("audio_vec norm:", np.linalg.norm(audio_vec) if audio_vec is not None else None)

    # Blend quarterly report embeddings into the transcript text vector
    if report_vec is not None and text_vec is not None:
        text_vec = (text_vec + report_vec) / 2.0
    elif report_vec is not None:
        text_vec = report_vec

    if text_vec is None and audio_vec is None:
        raise ValueError("No usable features could be extracted from the provided inputs.")

    # --- Multi-horizon inference ---
    horizon_labels = {4: "4Q Ahead", 8: "8Q Ahead", 16: "16Q Ahead"}
    distribution = []
    for k in K_HORIZONS:
        prob = _predict_horizon(k, text_vec, audio_vec)
        distribution.append(DistributionPoint(horizon_labels[k], round(prob, 4)))

    peak  = max(pt.probability for pt in distribution)
    score = int(np.clip(peak * 100, 0, 100))

    modality = (
        "late_fusion" if (text_vec is not None and audio_vec is not None)
        else ("audio" if audio_vec is not None else "text")
    )
    feat_modality = "audio" if audio_vec is not None else "text"

    result = AnalysisResult(
        overallRiskScore=score,
        riskLevel=score_to_risk_level(score),
        distribution=distribution,
        topFeatures=_get_top_features(_PRIMARY_K, feat_modality),
        modelInfo=_get_model_info(_PRIMARY_K, modality),
    )
    return result.to_dict()

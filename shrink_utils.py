#!/usr/bin/env python3
"""
Shared utilities for the MAEC call compression pipeline.

Reusable helpers for instance discovery, file I/O, missing-value handling,
and temporal feature computation.
"""

import csv
import re
from datetime import date
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

UNDEFINED_SENTINEL = "--undefined--"
STAT_NAMES = ["mean", "std", "max", "min", "first", "last", "slope"]

# ---------------------------------------------------------------------------
# Instance Discovery
# ---------------------------------------------------------------------------

_INSTANCE_RE = re.compile(r"(\d{8})_([A-Z0-9.\-]+)", re.IGNORECASE)


def parse_instance_folder(name: str) -> Optional[Tuple[date, str]]:
    """Parse YYYYMMDD_TICKER folder name into (date, ticker).

    Returns None if the name doesn't match the expected pattern.
    """
    m = _INSTANCE_RE.fullmatch(name.strip())
    if m is None:
        return None
    try:
        d = date(int(m.group(1)[:4]), int(m.group(1)[4:6]), int(m.group(1)[6:8]))
    except ValueError:
        return None
    return d, m.group(2).upper()


def discover_instances(maec_root: Path) -> List[Path]:
    """Return sorted list of valid MAEC instance directories."""
    return sorted(
        d for d in maec_root.iterdir()
        if d.is_dir() and parse_instance_folder(d.name) is not None
    )


def extract_identifiers(inst_dir: Path) -> Tuple[str, str, str]:
    """Extract (instance_id, ticker, call_date_iso) from instance folder name.

    instance_id: full folder name e.g. "20150225_LMAT"
    ticker:      e.g. "LMAT"
    call_date:   e.g. "2015-02-25"
    """
    parsed = parse_instance_folder(inst_dir.name)
    assert parsed is not None, f"Bad instance folder name: {inst_dir.name}"
    d, ticker = parsed
    return inst_dir.name, ticker, d.isoformat()


# ---------------------------------------------------------------------------
# File I/O
# ---------------------------------------------------------------------------


def load_features_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load features.csv, replacing '--undefined--' with NaN.

    Returns:
        header: list of column names (length 29 typically)
        data:   numpy float64 array of shape (N, 29), NaN where undefined
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows: List[List[float]] = []
        for row in reader:
            parsed: List[float] = []
            for val in row:
                val = val.strip()
                if val == UNDEFINED_SENTINEL or val == "":
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(val))
                    except ValueError:
                        parsed.append(np.nan)
            rows.append(parsed)
    return header, np.array(rows, dtype=np.float64)


def load_sbert_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load sbert.csv as float64 numpy array.

    Returns:
        header: list of column names (e0..e383 typically)
        data:   numpy float64 array of shape (N, 384)
    """
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        rows: List[List[float]] = []
        for row in reader:
            parsed: List[float] = []
            for val in row:
                val = val.strip()
                if val == UNDEFINED_SENTINEL or val == "":
                    parsed.append(np.nan)
                else:
                    try:
                        parsed.append(float(val))
                    except ValueError:
                        parsed.append(np.nan)
            rows.append(parsed)
    return header, np.array(rows, dtype=np.float64)


def load_sentences(text_path: Path) -> List[str]:
    """Load text.txt, one sentence per line. Returns list of strings."""
    with text_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f]


def write_shrink_csv(path: Path, header: List[str], vector: np.ndarray) -> None:
    """Write a single-row shrink CSV with header.

    vector: 1-D numpy array of length D.
    header: list of D column names.
    """
    assert vector.ndim == 1, f"Expected 1-D vector, got shape {vector.shape}"
    assert len(header) == len(vector), (
        f"Header length {len(header)} != vector length {len(vector)}"
    )
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerow([f"{v:.8f}" for v in vector])


def load_shrink_csv(path: Path) -> Tuple[List[str], np.ndarray]:
    """Load a single-row shrink CSV back into (header, 1D array)."""
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = [h.strip() for h in next(reader)]
        values = [float(v) for v in next(reader)]
    return header, np.array(values, dtype=np.float64)


# ---------------------------------------------------------------------------
# Missing Value Handling
# ---------------------------------------------------------------------------


def impute_nan_to_zero(data: np.ndarray) -> np.ndarray:
    """Replace NaN values with 0.0. Returns a copy."""
    out = data.copy()
    out[np.isnan(out)] = 0.0
    return out


def impute_nan_to_column_mean(data: np.ndarray) -> np.ndarray:
    """Replace NaN values with per-column mean (computed ignoring NaN).

    If an entire column is NaN, fills with 0.0. Returns a copy.
    """
    out = data.copy()
    for col_idx in range(out.shape[1]):
        col = out[:, col_idx]
        mask = np.isnan(col)
        if mask.any():
            col_mean = np.nanmean(col)
            if np.isnan(col_mean):
                col_mean = 0.0
            col[mask] = col_mean
    return out


# ---------------------------------------------------------------------------
# Temporal Feature Computation
# ---------------------------------------------------------------------------


def compute_temporal_stats(matrix: np.ndarray) -> Tuple[np.ndarray, List[str]]:
    """Compute temporal summary statistics for a sequence matrix.

    Args:
        matrix: shape (T, D) where T = sequence length, D = feature dim

    Returns:
        vector: 1-D array of length D*7 containing, in order:
            [mean_d0..mean_dD-1, std_d0..std_dD-1, max_d0..max_dD-1,
             min_d0..min_dD-1, first_d0..first_dD-1, last_d0..last_dD-1,
             slope_d0..slope_dD-1]
        stat_names: list of 7 stat name strings

    For T=1: std=0, slope=0, first=last=mean.
    For T=0: returns all zeros.

    Slope is computed via least-squares linear fit on normalized positions
    [0, 1] to capture temporal trend direction and magnitude.
    """
    T, D = matrix.shape

    if T == 0:
        return np.zeros(D * len(STAT_NAMES), dtype=np.float64), STAT_NAMES

    if T == 1:
        val = matrix[0]  # shape (D,)
        # Replace any remaining NaN with 0 for safety
        val = np.where(np.isnan(val), 0.0, val)
        return np.concatenate([
            val,            # mean
            np.zeros(D),    # std
            val,            # max
            val,            # min
            val,            # first
            val,            # last
            np.zeros(D),    # slope
        ]), STAT_NAMES

    # T >= 2
    stat_mean = np.nanmean(matrix, axis=0)            # (D,)
    stat_std = np.nanstd(matrix, axis=0, ddof=0)      # (D,) population std
    stat_max = np.nanmax(matrix, axis=0)               # (D,)
    stat_min = np.nanmin(matrix, axis=0)               # (D,)
    stat_first = matrix[0].copy()                      # (D,)
    stat_last = matrix[-1].copy()                      # (D,)

    # Slope: linear regression coefficient over normalized time [0, 1]
    # slope_j = cov(t, x_j) / var(t) for each column j
    t = np.linspace(0.0, 1.0, T)               # (T,)
    t_mean = t.mean()
    t_centered = t - t_mean                     # (T,)
    var_t = np.dot(t_centered, t_centered)      # scalar, > 0 since T >= 2

    # Replace NaN in matrix with column means for slope computation
    matrix_for_slope = matrix.copy()
    nan_mask = np.isnan(matrix_for_slope)
    if nan_mask.any():
        col_means = np.nanmean(matrix_for_slope, axis=0)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)
        inds = np.where(nan_mask)
        matrix_for_slope[inds] = np.take(col_means, inds[1])

    matrix_centered = matrix_for_slope - matrix_for_slope.mean(axis=0, keepdims=True)
    cov_tx = t_centered @ matrix_centered       # (D,)
    stat_slope = cov_tx / var_t                 # (D,)

    # Replace any NaN in other stats with 0
    for arr in [stat_mean, stat_std, stat_max, stat_min, stat_first, stat_last]:
        arr[np.isnan(arr)] = 0.0

    return np.concatenate([
        stat_mean, stat_std, stat_max, stat_min,
        stat_first, stat_last, stat_slope,
    ]), STAT_NAMES


def make_shrink_header(feature_names: List[str], stat_names: List[str]) -> List[str]:
    """Generate column names for shrink vector.

    Format: {stat}_{feature_name}
    Ordering: stats are the outer loop (all features for mean, then all for std, etc.)
    e.g., mean_e0, mean_e1, ..., mean_e383, std_e0, ..., slope_e383
    """
    header: List[str] = []
    for stat in stat_names:
        for feat in feature_names:
            header.append(f"{stat}_{feat}")
    return header

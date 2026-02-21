#!/usr/bin/env python3
"""
Aggregate per-call shrink vectors into global call-level matrices.

Reads text_shrink.csv and audio_shrink.csv from each MAEC instance,
optionally applies global z-score normalization and PCA dimensionality
reduction, then writes combined matrices to data/processed/.

Outputs:
  data/processed/text_call_matrix.csv   (~3443 rows x identifier cols + feature cols)
  data/processed/audio_call_matrix.csv  (~3443 rows x identifier cols + feature cols)
  data/processed/aggregation_log.json   (metadata and statistics)

Usage:
    python3 aggregate_calls.py [--normalize] [--pca-text 300] [--pca-audio 30]
"""

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from shrink_utils import (
    discover_instances,
    extract_identifiers,
    load_shrink_csv,
)


def collect_shrink_vectors(
    instance_dirs: List[Path],
    shrink_filename: str,
) -> Tuple[List[str], List[str], np.ndarray, List[str]]:
    """Load shrink vectors from all instances.

    Args:
        instance_dirs:   list of instance directory paths
        shrink_filename: "text_shrink.csv" or "audio_shrink.csv"

    Returns:
        instance_ids: list of instance_id strings (len = N_found)
        header:       list of dimension column names
        matrix:       numpy array of shape (N_found, D)
        skipped:      list of instance_ids where the file was missing
    """
    instance_ids: List[str] = []
    rows: List[np.ndarray] = []
    skipped: List[str] = []
    header: Optional[List[str]] = None

    for inst_dir in instance_dirs:
        shrink_path = inst_dir / shrink_filename
        inst_id = inst_dir.name

        if not shrink_path.exists():
            skipped.append(inst_id)
            continue

        try:
            h, vec = load_shrink_csv(shrink_path)
        except Exception as e:
            print(f"[WARN] {inst_id}: failed to load {shrink_filename}: {e}")
            skipped.append(inst_id)
            continue

        if header is None:
            header = h
        elif len(h) != len(header):
            print(
                f"[WARN] {inst_id}: dimension mismatch "
                f"(expected {len(header)}, got {len(h)}). Skipping."
            )
            skipped.append(inst_id)
            continue

        instance_ids.append(inst_id)
        rows.append(vec)

    if not rows:
        return [], header or [], np.empty((0, 0)), skipped

    matrix = np.vstack(rows)
    return instance_ids, header or [], matrix, skipped


def global_zscore(
    matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply global z-score normalization: (x - mu) / sigma per column.

    Columns with std=0 are left as-is (0 after mean subtraction).

    Returns:
        normalized: z-scored matrix
        means:      per-column means
        stds:       per-column stds
    """
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0, ddof=0)
    # Avoid division by zero
    safe_stds = stds.copy()
    safe_stds[safe_stds == 0] = 1.0
    normalized = (matrix - means) / safe_stds
    return normalized, means, stds


def apply_pca(
    matrix: np.ndarray,
    n_components: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply PCA to reduce dimensionality using numpy SVD.

    Centers data before SVD. No sklearn dependency.

    Args:
        matrix:       shape (N, D_in)
        n_components: target dimensionality

    Returns:
        reduced:                 shape (N, n_components)
        components:              shape (n_components, D_in) -- principal axes
        explained_variance_ratio: 1-D array of length n_components
    """
    N, D = matrix.shape
    n_components = min(n_components, N, D)

    mean = matrix.mean(axis=0)
    X = matrix - mean

    # Economy SVD
    U, S, Vt = np.linalg.svd(X, full_matrices=False)

    components = Vt[:n_components]               # (n_components, D)
    reduced = X @ components.T                   # (N, n_components)

    explained_var = (S ** 2) / max(N - 1, 1)
    total_var = explained_var.sum()
    if total_var > 0:
        explained_ratio = explained_var[:n_components] / total_var
    else:
        explained_ratio = np.zeros(n_components)

    return reduced, components, explained_ratio


def write_call_matrix(
    path: Path,
    instance_ids: List[str],
    header: List[str],
    matrix: np.ndarray,
) -> None:
    """Write the global call matrix CSV.

    Columns: instance_id, ticker, call_date, then feature columns.
    Extracts ticker and call_date from instance_id string.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["instance_id", "ticker", "call_date"] + header)

        for i, inst_id in enumerate(instance_ids):
            # Parse YYYYMMDD_TICKER
            parts = inst_id.split("_", 1)
            date_str = parts[0]
            ticker = parts[1] if len(parts) > 1 else ""
            call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

            row_values = [f"{v:.8f}" for v in matrix[i]]
            writer.writerow([inst_id, ticker, call_date] + row_values)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Aggregate per-call shrink vectors into global matrices"
    )
    ap.add_argument(
        "--maec-root",
        default="data/MAEC_Dataset",
        help="Root directory containing MAEC instance folders.",
    )
    ap.add_argument(
        "--output-dir",
        default="data/processed",
        help="Directory for output matrices.",
    )
    ap.add_argument(
        "--pca-text",
        type=int,
        default=None,
        help="Target text dimensions after PCA (default: no PCA, keep all).",
    )
    ap.add_argument(
        "--pca-audio",
        type=int,
        default=None,
        help="Target audio dimensions after PCA (default: no PCA, keep all).",
    )
    ap.add_argument(
        "--normalize",
        action="store_true",
        help="Apply global z-score normalization before PCA.",
    )
    ap.add_argument(
        "--log-file",
        default=None,
        help="Path to save the aggregation log (default: <output-dir>/aggregation_log.json).",
    )
    args = ap.parse_args()

    maec_root = Path(args.maec_root)
    output_dir = Path(args.output_dir)
    log_path = Path(args.log_file) if args.log_file else output_dir / "aggregation_log.json"

    if not maec_root.is_dir():
        print(f"[ERR] MAEC root not found: {maec_root}")
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # Discover instances
    instance_dirs = discover_instances(maec_root)
    print(f"[INFO] Found {len(instance_dirs)} MAEC instance folders")

    # Collect text shrink vectors
    print("[INFO] Collecting text shrink vectors...")
    text_ids, text_header, text_matrix, text_skipped = collect_shrink_vectors(
        instance_dirs, "text_shrink.csv"
    )
    print(
        f"[INFO] Text: {len(text_ids)} loaded, {len(text_skipped)} skipped, "
        f"raw shape = {text_matrix.shape}"
    )

    # Collect audio shrink vectors
    print("[INFO] Collecting audio shrink vectors...")
    audio_ids, audio_header, audio_matrix, audio_skipped = collect_shrink_vectors(
        instance_dirs, "audio_shrink.csv"
    )
    print(
        f"[INFO] Audio: {len(audio_ids)} loaded, {len(audio_skipped)} skipped, "
        f"raw shape = {audio_matrix.shape}"
    )

    # Check alignment
    if text_ids != audio_ids:
        text_set = set(text_ids)
        audio_set = set(audio_ids)
        only_text = text_set - audio_set
        only_audio = audio_set - text_set
        if only_text or only_audio:
            print(
                f"[WARN] Instance mismatch: {len(only_text)} text-only, "
                f"{len(only_audio)} audio-only"
            )

    log: Dict = {
        "started_at": datetime.now().isoformat(),
        "instances_found": len(instance_dirs),
        "text_instances_loaded": len(text_ids),
        "text_instances_skipped": len(text_skipped),
        "audio_instances_loaded": len(audio_ids),
        "audio_instances_skipped": len(audio_skipped),
        "text_raw_dims": text_matrix.shape[1] if text_matrix.size else 0,
        "audio_raw_dims": audio_matrix.shape[1] if audio_matrix.size else 0,
        "normalize": args.normalize,
    }

    # Check for NaN/Inf before processing
    for name, mat in [("text", text_matrix), ("audio", audio_matrix)]:
        if mat.size > 0:
            n_nan = np.isnan(mat).sum()
            n_inf = np.isinf(mat).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"[WARN] {name} matrix has {n_nan} NaN and {n_inf} Inf values")

    # Optional normalization
    if args.normalize and text_matrix.size > 0:
        print("[INFO] Applying global z-score normalization to text...")
        text_matrix, text_means, text_stds = global_zscore(text_matrix)
        log["text_norm_stats_saved"] = True

    if args.normalize and audio_matrix.size > 0:
        print("[INFO] Applying global z-score normalization to audio...")
        audio_matrix, audio_means, audio_stds = global_zscore(audio_matrix)
        log["audio_norm_stats_saved"] = True

    # Optional PCA
    if args.pca_text is not None and text_matrix.size > 0:
        n_comp = args.pca_text
        print(f"[INFO] Applying PCA to text: {text_matrix.shape[1]} -> {n_comp} dims...")
        text_matrix, text_components, text_var_ratio = apply_pca(text_matrix, n_comp)
        actual_dims = text_matrix.shape[1]
        text_header = [f"pc{i}" for i in range(actual_dims)]
        cumulative_var = float(text_var_ratio.sum())
        print(f"[INFO] Text PCA: {actual_dims} components, "
              f"cumulative explained variance = {cumulative_var:.4f}")
        log["pca_text_components"] = actual_dims
        log["pca_text_explained_variance"] = cumulative_var

        # Save PCA components
        pca_path = output_dir / "text_pca_components.npy"
        np.save(pca_path, text_components)
        print(f"[INFO] Saved text PCA components to {pca_path}")

    if args.pca_audio is not None and audio_matrix.size > 0:
        n_comp = args.pca_audio
        print(f"[INFO] Applying PCA to audio: {audio_matrix.shape[1]} -> {n_comp} dims...")
        audio_matrix, audio_components, audio_var_ratio = apply_pca(audio_matrix, n_comp)
        actual_dims = audio_matrix.shape[1]
        audio_header = [f"pc{i}" for i in range(actual_dims)]
        cumulative_var = float(audio_var_ratio.sum())
        print(f"[INFO] Audio PCA: {actual_dims} components, "
              f"cumulative explained variance = {cumulative_var:.4f}")
        log["pca_audio_components"] = actual_dims
        log["pca_audio_explained_variance"] = cumulative_var

        # Save PCA components
        pca_path = output_dir / "audio_pca_components.npy"
        np.save(pca_path, audio_components)
        print(f"[INFO] Saved audio PCA components to {pca_path}")

    # Write output matrices
    text_out = output_dir / "text_call_matrix.csv"
    audio_out = output_dir / "audio_call_matrix.csv"

    if text_matrix.size > 0:
        print(f"[INFO] Writing {text_out} ...")
        write_call_matrix(text_out, text_ids, text_header, text_matrix)
        log["text_output_shape"] = list(text_matrix.shape)
    else:
        print("[WARN] No text data to write.")

    if audio_matrix.size > 0:
        print(f"[INFO] Writing {audio_out} ...")
        write_call_matrix(audio_out, audio_ids, audio_header, audio_matrix)
        log["audio_output_shape"] = list(audio_matrix.shape)
    else:
        print("[WARN] No audio data to write.")

    elapsed = time.time() - t_start
    log["completed_at"] = datetime.now().isoformat()
    log["elapsed_seconds"] = round(elapsed, 1)

    # Save log
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    # Final output verification
    for name, mat in [("text", text_matrix), ("audio", audio_matrix)]:
        if mat.size > 0:
            n_nan = np.isnan(mat).sum()
            n_inf = np.isinf(mat).sum()
            if n_nan > 0 or n_inf > 0:
                print(f"[WARN] Final {name} matrix has {n_nan} NaN and {n_inf} Inf")
            else:
                print(f"[OK]   {name} matrix: no NaN or Inf values")

    print(f"\n{'='*60}")
    print("AGGREGATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Instances found:    {len(instance_dirs)}")
    print(f"  Text loaded:        {len(text_ids)} ({text_matrix.shape if text_matrix.size else 'empty'})")
    print(f"  Audio loaded:       {len(audio_ids)} ({audio_matrix.shape if audio_matrix.size else 'empty'})")
    print(f"  Normalize:          {args.normalize}")
    print(f"  PCA text:           {args.pca_text or 'none'}")
    print(f"  PCA audio:          {args.pca_audio or 'none'}")
    print(f"  Output dir:         {output_dir}")
    print(f"  Elapsed:            {elapsed:.1f}s")
    print(f"  Log saved:          {log_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

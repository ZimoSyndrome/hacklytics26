#!/usr/bin/env python3
"""
Compress MAEC earnings call sequences into per-call shrink vectors.

For each MAEC instance folder (data/MAEC_Dataset/YYYYMMDD_TICKER):
  1. Load sbert.csv   (N x 384 SBERT embeddings)
     -> compute 7 temporal stats per column -> text_shrink.csv  (1 x 2688)
  2. Load features.csv (N x 29 audio features, --undefined-- -> NaN -> impute)
     -> compute 7 temporal stats per column -> audio_shrink.csv (1 x 203)

Compression method: deterministic statistical temporal summaries.
For each feature column, computes [mean, std, max, min, first, last, slope]
across the sentence sequence. No neural network, no training.

Usage:
    python3 compress_calls.py [--maec-root data/MAEC_Dataset] [--force] [--impute zero]
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from shrink_utils import (
    discover_instances,
    load_features_csv,
    load_sbert_csv,
    impute_nan_to_zero,
    impute_nan_to_column_mean,
    compute_temporal_stats,
    make_shrink_header,
    write_shrink_csv,
)


def compress_one_instance(
    inst_dir: Path,
    impute_method: str = "zero",
    min_sentences: int = 1,
) -> Dict[str, Any]:
    """Compress a single instance folder.

    Args:
        inst_dir:       path to YYYYMMDD_TICKER directory
        impute_method:  "zero" or "column_mean" for features.csv missing values
        min_sentences:  skip instances with fewer sentences than this

    Returns:
        dict with keys:
            status:     "ok" | "skip" | "error"
            reason:     str (if skip/error)
            sentences:  int (if ok)
            text_dims:  int (if ok)
            audio_dims: int (if ok)
    """
    sbert_path = inst_dir / "sbert.csv"
    feat_path = inst_dir / "features.csv"
    text_shrink_path = inst_dir / "text_shrink.csv"
    audio_shrink_path = inst_dir / "audio_shrink.csv"

    # Check required files
    if not sbert_path.exists():
        return {"status": "skip", "reason": "missing sbert.csv"}
    if not feat_path.exists():
        return {"status": "skip", "reason": "missing features.csv"}

    # Load data
    try:
        sbert_header, sbert_matrix = load_sbert_csv(sbert_path)
        feat_header, feat_matrix = load_features_csv(feat_path)
    except Exception as e:
        return {"status": "error", "reason": f"load failed: {str(e)[:200]}"}

    n_sbert = sbert_matrix.shape[0]
    n_feat = feat_matrix.shape[0]

    # Validate row counts match
    if n_sbert != n_feat:
        return {
            "status": "skip",
            "reason": f"row mismatch: sbert={n_sbert}, features={n_feat}",
        }

    n_sentences = n_sbert

    if n_sentences < min_sentences:
        return {
            "status": "skip",
            "reason": f"only {n_sentences} sentences (min={min_sentences})",
        }

    # Impute NaN in audio features
    if impute_method == "zero":
        feat_matrix = impute_nan_to_zero(feat_matrix)
    elif impute_method == "column_mean":
        feat_matrix = impute_nan_to_column_mean(feat_matrix)

    # Also impute any NaN in sbert (shouldn't normally exist, but safety)
    if np.isnan(sbert_matrix).any():
        sbert_matrix = impute_nan_to_zero(sbert_matrix)

    # Compute temporal statistics
    text_vector, stat_names = compute_temporal_stats(sbert_matrix)
    audio_vector, _ = compute_temporal_stats(feat_matrix)

    # Generate headers
    text_header = make_shrink_header(sbert_header, stat_names)
    audio_header = make_shrink_header(feat_header, stat_names)

    # Write shrink CSVs
    write_shrink_csv(text_shrink_path, text_header, text_vector)
    write_shrink_csv(audio_shrink_path, audio_header, audio_vector)

    return {
        "status": "ok",
        "sentences": n_sentences,
        "text_dims": len(text_vector),
        "audio_dims": len(audio_vector),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Compress MAEC call sequences into per-call shrink vectors"
    )
    ap.add_argument(
        "--maec-root",
        default="data/MAEC_Dataset",
        help="Root directory containing MAEC instance folders.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-compress even if shrink CSVs already exist.",
    )
    ap.add_argument(
        "--impute",
        choices=["zero", "column_mean"],
        default="zero",
        help="Strategy for replacing --undefined-- in features.csv (default: zero).",
    )
    ap.add_argument(
        "--min-sentences",
        type=int,
        default=1,
        help="Skip instances with fewer than this many sentences (default: 1).",
    )
    ap.add_argument(
        "--log-file",
        default="data/compress_calls_log.json",
        help="Path to save the compression log.",
    )
    args = ap.parse_args()

    maec_root = Path(args.maec_root)
    if not maec_root.is_dir():
        print(f"[ERR] MAEC root not found: {maec_root}")
        return 1

    # Collect instance folders
    instance_dirs = discover_instances(maec_root)
    print(f"[INFO] Found {len(instance_dirs)} MAEC instance folders")

    # Processing stats
    log: Dict[str, Any] = {
        "method": "statistical_temporal_summaries",
        "stats_computed": ["mean", "std", "max", "min", "first", "last", "slope"],
        "impute_method": args.impute,
        "min_sentences": args.min_sentences,
        "started_at": datetime.now().isoformat(),
        "results": [],
    }
    success_count = 0
    skip_count = 0
    error_count = 0
    already_count = 0

    t_start = time.time()

    for idx, inst_dir in enumerate(instance_dirs):
        inst_name = inst_dir.name
        text_shrink_path = inst_dir / "text_shrink.csv"
        audio_shrink_path = inst_dir / "audio_shrink.csv"

        # Skip if already compressed
        if (
            text_shrink_path.exists()
            and audio_shrink_path.exists()
            and not args.force
        ):
            already_count += 1
            if (idx + 1) % 500 == 0:
                print(
                    f"[INFO] ({idx+1}/{len(instance_dirs)}) "
                    f"skip (already compressed)..."
                )
            continue

        # Compress
        result = compress_one_instance(
            inst_dir,
            impute_method=args.impute,
            min_sentences=args.min_sentences,
        )
        result["instance"] = inst_name

        if result["status"] == "ok":
            success_count += 1
        elif result["status"] == "skip":
            skip_count += 1
            print(f"[WARN] {inst_name}: {result['reason']}. Skipping.")
        else:
            error_count += 1
            print(f"[ERR]  {inst_name}: {result.get('reason', 'unknown error')}")

        log["results"].append(result)

        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(instance_dirs) - idx - 1) / rate
            print(
                f"[INFO] ({idx+1}/{len(instance_dirs)}) "
                f"compressed {inst_name} | "
                f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
            )

    elapsed_total = time.time() - t_start

    # Final stats
    log["completed_at"] = datetime.now().isoformat()
    log["elapsed_seconds"] = round(elapsed_total, 1)
    log["summary"] = {
        "total_folders": len(instance_dirs),
        "newly_compressed": success_count,
        "already_existed": already_count,
        "skipped": skip_count,
        "errors": error_count,
    }

    # Save log
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print("CALL COMPRESSION COMPLETE")
    print(f"{'='*60}")
    print(f"  Method:             statistical temporal summaries")
    print(f"  Stats:              mean, std, max, min, first, last, slope")
    print(f"  Impute:             {args.impute}")
    print(f"  Folders found:      {len(instance_dirs)}")
    print(f"  Newly compressed:   {success_count}")
    print(f"  Already existed:    {already_count}")
    print(f"  Skipped:            {skip_count}")
    print(f"  Errors:             {error_count}")
    print(f"  Elapsed:            {elapsed_total:.1f}s")
    print(f"  Log saved:          {log_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

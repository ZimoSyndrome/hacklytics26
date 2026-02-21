#!/usr/bin/env python3
"""
Encode each sentence in MAEC text.txt using Sentence-BERT and save embeddings to sbert.csv.

For every MAEC instance folder (data/MAEC_Dataset/YYYYMMDD_TICKER):
  1. Load text.txt   — one sentence per line
  2. Load features.csv — audio features with header; data rows must equal sentence count
  3. Encode sentences with a Sentence-BERT model (default: all-MiniLM-L6-v2)
  4. Save embeddings to sbert.csv with header e0,e1,...,eN-1  (same row order)

Skips folders with missing files or row-count mismatches; logs all issues.

Usage:
    python3 encode_sbert.py [--model all-MiniLM-L6-v2] [--batch-size 128]
"""

import argparse
import csv
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


def parse_instance_folder(name: str) -> bool:
    """Check if a folder name matches YYYYMMDD_TICKER."""
    return bool(re.fullmatch(r"\d{8}_[A-Z0-9.\-]+", name.strip().upper()))


def load_sentences(text_path: Path) -> List[str]:
    """Load sentences from text.txt, one per line. Returns list of strings."""
    with text_path.open("r", encoding="utf-8") as f:
        return [line.rstrip("\n\r") for line in f]


def count_csv_data_rows(csv_path: Path) -> int:
    """Count data rows in features.csv (excludes header)."""
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        return sum(1 for _ in reader)


def main():
    ap = argparse.ArgumentParser(description="Encode MAEC sentences with Sentence-BERT")
    ap.add_argument(
        "--maec-root",
        default="data/MAEC_Dataset",
        help="Root directory containing MAEC instance folders.",
    )
    ap.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-BERT model name (HuggingFace model hub ID).",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding (adjust based on GPU/CPU memory).",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if sbert.csv already exists.",
    )
    ap.add_argument(
        "--log-file",
        default="data/sbert_encoding_log.json",
        help="Path to save the encoding log.",
    )
    args = ap.parse_args()

    maec_root = Path(args.maec_root)
    if not maec_root.is_dir():
        print(f"[ERR] MAEC root not found: {maec_root}")
        return 1

    # Collect instance folders
    instance_dirs = sorted([
        d for d in maec_root.iterdir()
        if d.is_dir() and parse_instance_folder(d.name)
    ])
    print(f"[INFO] Found {len(instance_dirs)} MAEC instance folders")

    # Load model
    print(f"[INFO] Loading Sentence-BERT model: {args.model}")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(args.model)
    embedding_dim = model.get_sentence_embedding_dimension()
    print(f"[INFO] Embedding dimension: {embedding_dim}")

    # Header for sbert.csv
    header = [f"e{i}" for i in range(embedding_dim)]

    # Processing stats
    log = {
        "model": args.model,
        "embedding_dim": embedding_dim,
        "started_at": datetime.now().isoformat(),
        "results": [],
    }
    success_count = 0
    skip_count = 0
    error_count = 0
    total_sentences = 0

    t_start = time.time()

    for idx, inst_dir in enumerate(instance_dirs):
        inst_name = inst_dir.name
        text_path = inst_dir / "text.txt"
        feat_path = inst_dir / "features.csv"
        out_path = inst_dir / "sbert.csv"

        # Skip if already encoded
        if out_path.exists() and not args.force:
            success_count += 1
            if (idx + 1) % 200 == 0:
                print(f"[INFO] ({idx+1}/{len(instance_dirs)}) skip (already encoded)...")
            continue

        # Check for missing files
        if not text_path.exists():
            log["results"].append({"instance": inst_name, "status": "skip", "reason": "missing text.txt"})
            skip_count += 1
            continue
        if not feat_path.exists():
            log["results"].append({"instance": inst_name, "status": "skip", "reason": "missing features.csv"})
            skip_count += 1
            continue

        # Load sentences
        sentences = load_sentences(text_path)
        n_sentences = len(sentences)

        if n_sentences == 0:
            log["results"].append({"instance": inst_name, "status": "skip", "reason": "empty text.txt"})
            skip_count += 1
            continue

        # Check row alignment
        n_feat_rows = count_csv_data_rows(feat_path)
        if n_sentences != n_feat_rows:
            log["results"].append({
                "instance": inst_name,
                "status": "mismatch",
                "reason": f"text.txt has {n_sentences} lines, features.csv has {n_feat_rows} data rows",
            })
            error_count += 1
            print(f"[WARN] {inst_name}: row mismatch — text={n_sentences}, features={n_feat_rows}. Skipping.")
            continue

        # Encode
        try:
            embeddings = model.encode(
                sentences,
                batch_size=args.batch_size,
                show_progress_bar=False,
                normalize_embeddings=False,
            )
            # embeddings: ndarray of shape (n_sentences, embedding_dim)
            assert embeddings.shape == (n_sentences, embedding_dim), (
                f"Unexpected shape: {embeddings.shape}"
            )
        except Exception as e:
            log["results"].append({"instance": inst_name, "status": "error", "reason": str(e)[:200]})
            error_count += 1
            print(f"[ERR] {inst_name}: encoding failed — {e}")
            continue

        # Save to sbert.csv
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for row in embeddings:
                writer.writerow([f"{v:.6f}" for v in row])

        success_count += 1
        total_sentences += n_sentences
        log["results"].append({"instance": inst_name, "status": "ok", "sentences": n_sentences})

        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t_start
            rate = (idx + 1) / elapsed
            eta = (len(instance_dirs) - idx - 1) / rate
            print(
                f"[INFO] ({idx+1}/{len(instance_dirs)}) "
                f"encoded {inst_name} ({n_sentences} sentences) | "
                f"elapsed {elapsed:.0f}s, ETA {eta:.0f}s"
            )

    elapsed_total = time.time() - t_start

    # Final stats
    log["completed_at"] = datetime.now().isoformat()
    log["elapsed_seconds"] = round(elapsed_total, 1)
    log["summary"] = {
        "total_folders": len(instance_dirs),
        "success": success_count,
        "skipped": skip_count,
        "errors": error_count,
        "total_sentences_encoded": total_sentences,
    }

    # Save log
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    print(f"\n{'='*60}")
    print(f"SBERT ENCODING COMPLETE")
    print(f"{'='*60}")
    print(f"  Model:              {args.model}")
    print(f"  Embedding dim:      {embedding_dim}")
    print(f"  Folders processed:  {len(instance_dirs)}")
    print(f"  Success:            {success_count}")
    print(f"  Skipped:            {skip_count}")
    print(f"  Errors:             {error_count}")
    print(f"  Sentences encoded:  {total_sentences}")
    print(f"  Elapsed:            {elapsed_total:.1f}s")
    print(f"  Log saved:          {log_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

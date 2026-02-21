#!/usr/bin/env python3
"""
Diagnose EDGAR filing coverage for MAEC dataset instances.
Reports failure buckets and coverage statistics without downloading anything.

Usage:
    python3 diagnose_coverage.py
"""

import json
import os
import re
from collections import defaultdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def parse_instance_folder(name: str) -> Optional[Tuple[date, str]]:
    m = re.fullmatch(r"(\d{8})_([A-Z0-9.\-]+)", name.strip().upper())
    if not m:
        return None
    d = datetime.strptime(m.group(1), "%Y%m%d").date()
    return d, m.group(2)


def main():
    maec_root = Path("data/MAEC_Dataset")
    cache_root = Path(".sec_cache")
    reports_root = Path("data/reports/relevant_by_instance")
    overrides_path = Path("ticker_cik_overrides.json")

    # 1) Scan MAEC instances
    instances: List[Tuple[date, str]] = []
    for p in maec_root.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_instance_folder(p.name)
        if parsed:
            instances.append(parsed)

    tickers = sorted({t for _, t in instances})
    ticker_instances = defaultdict(list)
    for d, t in instances:
        ticker_instances[t].append(d)

    print(f"MAEC instances: {len(instances)} across {len(tickers)} tickers")

    # 2) Load CIK maps
    with (cache_root / "company_tickers.json").open() as f:
        data = json.load(f)
    current_map = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if t and isinstance(cik, int):
            current_map[t] = cik

    overrides = {}
    if overrides_path.exists():
        with overrides_path.open() as f:
            overrides = {k.upper(): int(v) for k, v in json.load(f).items()}

    full_map = {**current_map, **overrides}

    # 3) Classify each ticker
    bucket_no_cik = []
    bucket_no_cache = []
    bucket_no_relevant = []
    bucket_downloaded_empty = []
    bucket_success = []

    submissions_dir = cache_root / "submissions"

    for t in tickers:
        cik = full_map.get(t)
        if not cik:
            bucket_no_cik.append(t)
            continue

        # Check submissions cache
        cik10 = str(cik).zfill(10)
        cache_path = submissions_dir / f"CIK{cik10}.json"
        if not cache_path.exists():
            bucket_no_cache.append(t)
            continue

        # Check if any download exists
        has_files = False
        for d in ticker_instances[t]:
            inst_key = f"{d.strftime('%Y%m%d')}_{t}"
            inst_dir = reports_root / inst_key
            if inst_dir.exists():
                contents = [f for f in inst_dir.iterdir() if not f.name.startswith(".")]
                if contents:
                    has_files = True
                    break

        if has_files:
            bucket_success.append(t)
        else:
            # Check if the filings cover the MAEC period
            with cache_path.open() as f:
                sub = json.load(f)
            recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
            dates = recent.get("filingDate", []) or []
            forms = recent.get("form", []) or []
            oldest_relevant = None
            for fd, frm in zip(dates, forms):
                if frm in ("10-K", "10-Q"):
                    oldest_relevant = fd
            has_older_pages = bool((sub.get("filings", {}) or {}).get("files", []))

            earliest_maec = min(ticker_instances[t])
            if oldest_relevant and oldest_relevant > earliest_maec.isoformat() and has_older_pages:
                bucket_no_relevant.append((t, f"recent only to {oldest_relevant}, MAEC from {earliest_maec}, has older pages"))
            else:
                bucket_downloaded_empty.append((t, f"oldest_recent={oldest_relevant}, MAEC from {earliest_maec}"))

    # Print report
    print(f"\n{'='*70}")
    print("COVERAGE DIAGNOSIS")
    print(f"{'='*70}")
    print(f"✅ Success (has downloaded files):     {len(bucket_success)}")
    print(f"❌ No CIK mapping:                     {len(bucket_no_cik)}")
    print(f"❌ CIK found, no cached submissions:   {len(bucket_no_cache)}")
    print(f"❌ CIK+cache, recent too new (needs pagination): {len(bucket_no_relevant)}")
    print(f"❌ CIK+cache, other failure:            {len(bucket_downloaded_empty)}")
    print(f"{'='*70}")

    total = len(tickers)
    current_pct = len(bucket_success) / total * 100
    potential_pct = (len(bucket_success) + len(bucket_no_relevant) + len(bucket_no_cache) + len(bucket_downloaded_empty)) / total * 100
    print(f"\nCurrent coverage:   {len(bucket_success)}/{total} ({current_pct:.1f}%)")
    print(f"After pagination fix potential: +{len(bucket_no_relevant)} tickers")
    print(f"After CIK override fix potential: +{len(bucket_no_cik)} tickers (if all resolved)")

    if bucket_no_cik:
        print(f"\nNo CIK ({len(bucket_no_cik)} tickers, first 20):")
        print(f"  {', '.join(bucket_no_cik[:20])}")

    if bucket_no_relevant:
        print(f"\nNeeds pagination ({len(bucket_no_relevant)} tickers, first 10):")
        for t, detail in bucket_no_relevant[:10]:
            print(f"  {t}: {detail}")

    if bucket_downloaded_empty:
        print(f"\nOther failures ({len(bucket_downloaded_empty)} tickers, first 10):")
        for t, detail in bucket_downloaded_empty[:10]:
            print(f"  {t}: {detail}")


if __name__ == "__main__":
    main()

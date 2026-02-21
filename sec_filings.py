#!/usr/bin/env python3
"""
Download SEC EDGAR 10-K / 10-Q filings for companies present in a local MAEC repo.

What it does
1) Scans MAEC instances under: hacklytics26/data/MAEC_Dataset/
   where each instance folder is named like: YYYYMMDD_TICKER  (e.g., 20150225_LMAT)

2) For each (date, ticker) instance:
   - downloads the most recent 10-Q with filingDate <= instance date
   - downloads the most recent 10-K with filingDate <= instance date
   (these are saved under output/relevant_by_instance/...)

3) For each ticker:
   - downloads ALL historical 10-K and 10-Q filings back to 1990-01-01 (inclusive)
   (saved under output/historical_all/...)

Notes / constraints
- Some tickers won’t map cleanly to a CIK (mergers, ticker changes). We log and skip.
- Not every company has filings back to 1990 (many IPO’d later).
- SEC requires a real User-Agent identifying you. Set --user-agent accordingly.
- Respect SEC rate limits; this script throttles requests.

Tested assumptions
- Uses SEC’s public endpoints:
  - company tickers mapping: https://www.sec.gov/files/company_tickers.json
  - company submissions:       https://data.sec.gov/submissions/CIK##########.json
  - filing docs:              https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_nodashes}/{primaryDocument}
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


SEC_TICKERS_JSON = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_ARCHIVES_DOC = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_nodash}/{primary_doc}"

FORMS_WANTED = {"10-K", "10-Q"}  # you can add "10-K/A", "10-Q/A" if you want
START_DATE = date(1990, 1, 1)


@dataclass(frozen=True)
class Filing:
    form: str
    filing_date: date
    accession_number: str
    primary_document: str


def parse_instance_folder(name: str) -> Optional[Tuple[date, str]]:
    """
    Parse folder name like YYYYMMDD_TICKER -> (date, ticker).
    Returns None if not matching.
    """
    m = re.fullmatch(r"(\d{8})_([A-Z0-9.\-]+)", name.strip().upper())
    if not m:
        return None
    d = datetime.strptime(m.group(1), "%Y%m%d").date()
    ticker = m.group(2)
    return d, ticker


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


class SecClient:
    def __init__(self, user_agent: str, sleep_s: float = 0.2, timeout_s: int = 30):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a descriptive User-Agent with contact info. "
                "Pass e.g. --user-agent 'Hacklytics26 Team (your.email@domain.com)'"
            )
        # Do NOT set a fixed Host header; requests will set the correct Host for
        # each domain (e.g., www.sec.gov vs data.sec.gov). A fixed Host can cause
        # 404/403 when calling data.sec.gov endpoints.
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.sleep_s = sleep_s
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def get_json(self, url: str) -> dict:
        time.sleep(self.sleep_s)
        r = self.session.get(url, headers=self.headers, timeout=self.timeout_s)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            status = r.status_code
            snippet = (r.text or "")[:200].replace("\n", " ")
            raise requests.HTTPError(f"HTTP {status} for {url}. Response starts: {snippet}") from e
        return r.json()

    def download_file(self, url: str, out_path: Path) -> None:
        time.sleep(self.sleep_s)
        r = self.session.get(url, headers=self.headers, timeout=self.timeout_s, stream=True)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            status = r.status_code
            # For binary downloads, r.text may be empty; include headers only.
            raise requests.HTTPError(f"HTTP {status} for {url}. Headers: {dict(r.headers)}") from e
        ensure_dir(out_path.parent)
        with out_path.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def load_or_fetch_ticker_map(sec: SecClient, cache_path: Path) -> Dict[str, int]:
    """
    Returns mapping: TICKER -> CIK (int)
    Uses a local cache file if present to avoid repeated SEC calls.
    """
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = sec.get_json(SEC_TICKERS_JSON)
        ensure_dir(cache_path.parent)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    # company_tickers.json is keyed by integer-like strings: {"0": {"cik_str":..., "ticker":...}, ...}
    out: Dict[str, int] = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if t and isinstance(cik, int):
            out[t] = cik
    return out


def cik_to_10(cik_int: int) -> str:
    return str(cik_int).zfill(10)


def fetch_company_filings(sec: SecClient, cik_int: int, cache_dir: Path) -> List[Filing]:
    """
    Fetches company submissions JSON and extracts filings list.
    Caches the submissions JSON locally.
    """
    cik10 = cik_to_10(cik_int)
    cache_path = cache_dir / f"CIK{cik10}.json"
    if cache_path.exists():
        with cache_path.open("r", encoding="utf-8") as f:
            sub = json.load(f)
    else:
        url = SEC_SUBMISSIONS.format(cik10=cik10)
        sub = sec.get_json(url)
        ensure_dir(cache_path.parent)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(sub, f)

    # Recent filings are in: filings.recent.*
    recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
    forms = recent.get("form", []) or []
    dates = recent.get("filingDate", []) or []
    accs = recent.get("accessionNumber", []) or []
    prims = recent.get("primaryDocument", []) or []

    n = min(len(forms), len(dates), len(accs), len(prims))
    out: List[Filing] = []
    for i in range(n):
        form = str(forms[i]).strip()
        if form not in FORMS_WANTED:
            continue
        try:
            fd = datetime.strptime(dates[i], "%Y-%m-%d").date()
        except Exception:
            continue
        out.append(
            Filing(
                form=form,
                filing_date=fd,
                accession_number=str(accs[i]).strip(),
                primary_document=str(prims[i]).strip(),
            )
        )

    # Sort newest first
    out.sort(key=lambda x: x.filing_date, reverse=True)
    return out


def accession_no_nodashes(acc_no: str) -> str:
    return acc_no.replace("-", "")


def choose_latest_on_or_before(filings: List[Filing], form: str, asof: date) -> Optional[Filing]:
    for f in filings:
        if f.form == form and f.filing_date <= asof:
            return f
    return None


def build_doc_url(cik_int: int, filing: Filing) -> str:
    return SEC_ARCHIVES_DOC.format(
        cik_int=cik_int,
        acc_no_nodash=accession_no_nodashes(filing.accession_number),
        primary_doc=filing.primary_document,
    )


def safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s)


def download_relevant_for_instance(
    sec: SecClient,
    ticker: str,
    cik_int: int,
    instance_date: date,
    filings: List[Filing],
    out_root: Path,
) -> bool:
    """
    Downloads the latest 10-Q and 10-K available at the time of the instance date.

    Returns True if at least one relevant filing (10-Q/10-K) was available and
    either downloaded successfully or already existed on disk.
    """
    inst_key = f"{instance_date.strftime('%Y%m%d')}_{ticker}"
    out_dir = out_root / "relevant_by_instance" / inst_key
    ensure_dir(out_dir)

    any_available = False

    for form in ("10-Q", "10-K"):
        chosen = choose_latest_on_or_before(filings, form, instance_date)
        if not chosen:
            print(f"[WARN] No {form} for {ticker} (CIK {cik_int}) on/before {instance_date} for {inst_key}")
            continue

        url = build_doc_url(cik_int, chosen)
        fname = f"{ticker}_{form}_{chosen.filing_date.isoformat()}_{safe_name(chosen.primary_document)}"
        out_path = out_dir / fname

        # If already present, count as available
        if out_path.exists():
            any_available = True
            continue

        try:
            sec.download_file(url, out_path)
            any_available = True
            print(f"[OK] {inst_key} -> {form} {chosen.filing_date} saved: {out_path}")
        except Exception as e:
            print(f"[ERR] Failed download {url} -> {out_path}: {e}")

    return any_available


def download_historical_all(
    sec: SecClient,
    ticker: str,
    cik_int: int,
    filings: List[Filing],
    out_root: Path,
    start_date: date = START_DATE,
) -> None:
    """
    Downloads ALL 10-K and 10-Q filings from start_date onwards.
    """
    out_dir = out_root / "historical_all" / ticker
    ensure_dir(out_dir)

    for f in filings:
        if f.filing_date < start_date:
            continue
        # organise by year/form for convenience
        year_dir = out_dir / str(f.filing_date.year) / f.form
        ensure_dir(year_dir)

        url = build_doc_url(cik_int, f)
        fname = f"{ticker}_{f.form}_{f.filing_date.isoformat()}_{safe_name(f.primary_document)}"
        out_path = year_dir / fname
        if out_path.exists():
            continue

        try:
            sec.download_file(url, out_path)
            print(f"[OK] {ticker} {f.form} {f.filing_date} saved: {out_path}")
        except Exception as e:
            print(f"[ERR] Failed download {url} -> {out_path}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--maec-root",
        default="hacklytics26/data/MAEC_Dataset",
        help="Path to MAEC instances folder containing YYYYMMDD_TICKER subfolders.",
    )
    ap.add_argument(
        "--out",
        default="data/reports",
        help="Output directory for downloaded filings.",
    )
    ap.add_argument(
        "--cache",
        default=".sec_cache",
        help="Cache directory for SEC JSON responses (ticker map + submissions).",
    )
    ap.add_argument(
        "--user-agent",
        required=True,
        help="Required by SEC. Example: 'Hacklytics26 Team (your.email@domain.com)'",
    )
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Seconds to sleep between SEC requests (throttle). Increase if you get 429/403.",
    )
    ap.add_argument(
        "--download-historical",
        action="store_true",
        help="If set, also downloads ALL 10-K/10-Q back to 1990 for each MAEC ticker.",
    )
    ap.add_argument(
        "--start-date",
        default="1990-01-01",
        help="Earliest filingDate to download for historical mode (YYYY-MM-DD).",
    )
    args = ap.parse_args()

    maec_root = Path(args.maec_root)
    out_root = Path(args.out)
    cache_root = Path(args.cache)
    ensure_dir(out_root)
    ensure_dir(cache_root)

    try:
        start_dt = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    except Exception:
        print("[ERR] --start-date must be YYYY-MM-DD")
        return 2

    sec = SecClient(user_agent=args.user_agent, sleep_s=args.sleep)

    # 1) Scan MAEC instances
    if not maec_root.exists() or not maec_root.is_dir():
        print(f"[ERR] MAEC root not found: {maec_root}")
        return 2

    instances: List[Tuple[date, str, Path]] = []
    for p in maec_root.iterdir():
        if not p.is_dir():
            continue
        parsed = parse_instance_folder(p.name)
        if parsed:
            d, t = parsed
            instances.append((d, t, p))

    if not instances:
        print(f"[ERR] No instance folders found under {maec_root} matching YYYYMMDD_TICKER")
        return 2

    # 2) Build ticker set
    tickers = sorted({t for _, t, _ in instances})
    print(f"[INFO] Found {len(instances)} MAEC instances across {len(tickers)} tickers")

    # 3) Map tickers -> CIK
    ticker_map_cache = cache_root / "company_tickers.json"
    ticker_to_cik = load_or_fetch_ticker_map(sec, ticker_map_cache)

    # 4) For each ticker, fetch filings once, then process all instances
    submissions_cache_dir = cache_root / "submissions"
    ensure_dir(submissions_cache_dir)

    ticker_filings: Dict[str, List[Filing]] = {}

    for t in tickers:
        cik = ticker_to_cik.get(t)
        if not cik:
            print(f"[WARN] No CIK found for ticker {t}. Skipping.")
            continue
        try:
            filings = fetch_company_filings(sec, cik, submissions_cache_dir)
            ticker_filings[t] = filings
            print(f"[INFO] {t} (CIK {cik}) has {len(filings)} recent 10-K/10-Q filings in submissions feed")
        except Exception as e:
            print(f"[ERR] Failed fetching filings for {t} (CIK {cik}): {e}")

    # 5) Download relevant filings for each instance
    ticker_has_any_data: Dict[str, bool] = {t: False for t in tickers}

    for d, t, _ in sorted(instances, key=lambda x: (x[1], x[0])):
        cik = ticker_to_cik.get(t)
        filings = ticker_filings.get(t)
        if not cik or not filings:
            continue
        had_any = download_relevant_for_instance(sec, t, cik, d, filings, out_root)
        if had_any:
            ticker_has_any_data[t] = True

    total = len(tickers)
    success = sum(1 for t in tickers if ticker_has_any_data.get(t, False))
    pct = (success / total * 100.0) if total else 0.0
    failed_tickers = sorted([t for t in tickers if not ticker_has_any_data.get(t, False)])

    print(f"[STATS] Tickers with any relevant 10-K/10-Q pulled for at least one MAEC instance: {success}/{total} ({pct:.2f}%)")
    if failed_tickers:
        print("[STATS] Tickers with no relevant filings pulled:")
        print(", ".join(failed_tickers))
    else:
        print("[STATS] All tickers had at least one relevant filing pulled.")

    # 6) Optionally download all historical filings (as far back as available in submissions feed)
    # NOTE: SEC submissions "recent" list does not always include *all* filings back to 1990
    # for very active filers. For full-history you may need additional EDGAR index parsing.
    if args.download_historical:
        for t in tickers:
            cik = ticker_to_cik.get(t)
            filings = ticker_filings.get(t)
            if not cik or not filings:
                continue
            download_historical_all(sec, t, cik, filings, out_root, start_date=start_dt)

        print(
            "[NOTE] This script uses the company submissions 'recent' array for filings. "
            "If you need complete 1990+ coverage for very frequent filers beyond what's in 'recent', "
            "you must additionally parse EDGAR quarterly index files or use a search endpoint."
        )

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
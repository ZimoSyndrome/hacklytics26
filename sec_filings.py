#!/usr/bin/env python3
"""
Download SEC EDGAR 10-K / 10-Q filings for companies present in a local MAEC repo.

What it does
1) Scans MAEC instances under: data/MAEC_Dataset/
   where each instance folder is named like: YYYYMMDD_TICKER  (e.g., 20150225_LMAT)

2) For each (date, ticker) instance:
   - downloads the most recent 10-Q with filingDate <= instance date
   - downloads the most recent 10-K with filingDate <= instance date
   (these are saved under output/relevant_by_instance/...)

3) Optionally for each ticker:
   - downloads ALL historical 10-K and 10-Q filings back to 1990-01-01 (inclusive)
   (saved under output/historical_all/...)

Key fixes (v2)
- Fetches ALL filing pages via filings.files[] (older submissions pagination).
  Previously only filings.recent was read, missing filings before ~2019 for prolific filers.
- Supports ticker_cik_overrides.json for delisted/renamed tickers not in current SEC map.
- Retry logic with exponential backoff for transient HTTP errors (429, 503).
- Structured failure log written to data/reports/download_log.json.

Notes / constraints
- SEC requires a real User-Agent identifying you. Set --user-agent accordingly.
- Respect SEC rate limits; this script throttles requests (default 0.3s between calls).
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests


SEC_TICKERS_JSON = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik10}.json"
SEC_SUBMISSIONS_PAGE = "https://data.sec.gov/submissions/{page_name}"
SEC_ARCHIVES_DOC = "https://www.sec.gov/Archives/edgar/data/{cik_int}/{acc_no_nodash}/{primary_doc}"

FORMS_WANTED = {"10-K", "10-Q"}
START_DATE = date(1990, 1, 1)

MAX_RETRIES = 3
RETRY_BACKOFF = 2.0  # seconds base for exponential backoff


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
    def __init__(self, user_agent: str, sleep_s: float = 0.3, timeout_s: int = 30):
        if not user_agent or "@" not in user_agent:
            raise ValueError(
                "SEC requires a descriptive User-Agent with contact info. "
                "Pass e.g. --user-agent 'Hacklytics26 Team (your.email@domain.com)'"
            )
        self.headers = {
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        }
        self.sleep_s = sleep_s
        self.timeout_s = timeout_s
        self.session = requests.Session()

    def _request_with_retry(self, method: str, url: str, **kwargs) -> requests.Response:
        """Make an HTTP request with retry logic for transient failures."""
        last_exc = None
        for attempt in range(MAX_RETRIES):
            time.sleep(self.sleep_s)
            try:
                r = self.session.request(method, url, headers=self.headers,
                                         timeout=self.timeout_s, **kwargs)
                if r.status_code in (429, 503):
                    wait = RETRY_BACKOFF * (2 ** attempt)
                    print(f"[RETRY] HTTP {r.status_code} for {url}, waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                return r
            except requests.exceptions.ConnectionError as e:
                wait = RETRY_BACKOFF * (2 ** attempt)
                print(f"[RETRY] Connection error for {url}, waiting {wait:.1f}s (attempt {attempt+1}/{MAX_RETRIES}): {e}")
                time.sleep(wait)
                last_exc = e
            except requests.HTTPError as e:
                raise  # non-retryable HTTP errors
        raise last_exc or requests.HTTPError(f"Failed after {MAX_RETRIES} retries: {url}")

    def get_json(self, url: str) -> dict:
        r = self._request_with_retry("GET", url)
        return r.json()

    def download_file(self, url: str, out_path: Path) -> None:
        r = self._request_with_retry("GET", url, stream=True)
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

    out: Dict[str, int] = {}
    for _, rec in data.items():
        t = str(rec.get("ticker", "")).upper().strip()
        cik = rec.get("cik_str")
        if t and isinstance(cik, int):
            out[t] = cik
    return out


def load_cik_overrides(path: Path) -> Dict[str, int]:
    """Load ticker->CIK overrides from a JSON file (for delisted/renamed tickers)."""
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    # Accept both {ticker: cik_int} and {ticker: "cik_str"} formats
    out: Dict[str, int] = {}
    for k, v in data.items():
        out[k.upper().strip()] = int(v)
    return out


def cik_to_10(cik_int: int) -> str:
    return str(cik_int).zfill(10)


def _extract_filings_from_page(page_data: dict) -> List[Filing]:
    """Extract Filing objects from a submissions page (recent or older page)."""
    forms = page_data.get("form", []) or []
    dates = page_data.get("filingDate", []) or []
    accs = page_data.get("accessionNumber", []) or []
    prims = page_data.get("primaryDocument", []) or []

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
    return out


def fetch_company_filings(
    sec: SecClient, cik_int: int, cache_dir: Path, force_refetch: bool = False
) -> List[Filing]:
    """
    Fetches company submissions JSON and extracts filings list.
    Now fetches ALL pages (filings.recent + filings.files[]) for complete history.
    Caches the submissions JSON locally.
    """
    cik10 = cik_to_10(cik_int)
    cache_path = cache_dir / f"CIK{cik10}.json"

    if cache_path.exists() and not force_refetch:
        with cache_path.open("r", encoding="utf-8") as f:
            sub = json.load(f)
    else:
        url = SEC_SUBMISSIONS.format(cik10=cik10)
        sub = sec.get_json(url)
        ensure_dir(cache_path.parent)
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(sub, f)

    # --- Extract from filings.recent ---
    recent = (sub.get("filings", {}) or {}).get("recent", {}) or {}
    out = _extract_filings_from_page(recent)

    # --- Fetch older filing pages from filings.files[] ---
    files_list = (sub.get("filings", {}) or {}).get("files", []) or []
    for file_ref in files_list:
        page_name = file_ref.get("name", "") if isinstance(file_ref, dict) else ""
        if not page_name:
            continue

        page_cache = cache_dir / page_name
        if page_cache.exists() and not force_refetch:
            with page_cache.open("r", encoding="utf-8") as f:
                page_data = json.load(f)
        else:
            page_url = SEC_SUBMISSIONS_PAGE.format(page_name=page_name)
            try:
                page_data = sec.get_json(page_url)
                with page_cache.open("w", encoding="utf-8") as f:
                    json.dump(page_data, f)
            except Exception as e:
                print(f"[WARN] Failed to fetch older filings page {page_name} for CIK {cik_int}: {e}")
                continue

        out.extend(_extract_filings_from_page(page_data))

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
    download_log: list,
) -> bool:
    """
    Downloads the latest 10-Q and 10-K available at the time of the instance date.
    Returns True if at least one relevant filing was available and downloaded/existed.
    """
    inst_key = f"{instance_date.strftime('%Y%m%d')}_{ticker}"
    out_dir = out_root / "relevant_by_instance" / inst_key
    ensure_dir(out_dir)

    any_available = False

    for form in ("10-Q", "10-K"):
        chosen = choose_latest_on_or_before(filings, form, instance_date)
        if not chosen:
            download_log.append({
                "ticker": ticker, "instance": inst_key, "form": form,
                "status": "no_filing_found",
                "detail": f"No {form} with filingDate <= {instance_date}",
            })
            continue

        url = build_doc_url(cik_int, chosen)
        fname = f"{ticker}_{form}_{chosen.filing_date.isoformat()}_{safe_name(chosen.primary_document)}"
        out_path = out_dir / fname

        if out_path.exists():
            any_available = True
            continue

        try:
            sec.download_file(url, out_path)
            any_available = True
            download_log.append({
                "ticker": ticker, "instance": inst_key, "form": form,
                "status": "ok", "filing_date": str(chosen.filing_date),
                "accession": chosen.accession_number,
            })
            print(f"[OK] {inst_key} -> {form} {chosen.filing_date} saved: {out_path}")
        except Exception as e:
            download_log.append({
                "ticker": ticker, "instance": inst_key, "form": form,
                "status": "download_error", "detail": str(e)[:200],
            })
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
    """Downloads ALL 10-K and 10-Q filings from start_date onwards."""
    out_dir = out_root / "historical_all" / ticker
    ensure_dir(out_dir)

    for f in filings:
        if f.filing_date < start_date:
            continue
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
    ap = argparse.ArgumentParser(description="Download SEC EDGAR 10-K/10-Q filings for MAEC companies")
    ap.add_argument(
        "--maec-root",
        default="data/MAEC_Dataset",
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
        default=0.3,
        help="Seconds to sleep between SEC requests (throttle). Default 0.3s.",
    )
    ap.add_argument(
        "--overrides-file",
        default="ticker_cik_overrides.json",
        help="JSON file with ticker->CIK overrides for delisted/renamed tickers.",
    )
    ap.add_argument(
        "--force-refetch",
        action="store_true",
        help="Re-download SEC submissions JSON (ignoring cached versions). "
             "Use this after fixing pagination to pick up older filing pages.",
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

    # 3) Map tickers -> CIK (current map + overrides)
    ticker_map_cache = cache_root / "company_tickers.json"
    ticker_to_cik = load_or_fetch_ticker_map(sec, ticker_map_cache)

    overrides = load_cik_overrides(Path(args.overrides_file))
    if overrides:
        print(f"[INFO] Loaded {len(overrides)} CIK overrides from {args.overrides_file}")
        ticker_to_cik.update(overrides)

    # Report CIK mapping status
    mapped = [t for t in tickers if t in ticker_to_cik]
    unmapped = [t for t in tickers if t not in ticker_to_cik]
    print(f"[INFO] CIK mapped: {len(mapped)}/{len(tickers)}, unmapped: {len(unmapped)}")
    if unmapped:
        print(f"[WARN] Unmapped tickers ({len(unmapped)}): {', '.join(unmapped[:20])}{'...' if len(unmapped) > 20 else ''}")

    # 4) For each ticker, fetch filings once (now with full pagination)
    submissions_cache_dir = cache_root / "submissions"
    ensure_dir(submissions_cache_dir)

    ticker_filings: Dict[str, List[Filing]] = {}

    for i, t in enumerate(mapped):
        cik = ticker_to_cik[t]
        try:
            filings = fetch_company_filings(
                sec, cik, submissions_cache_dir, force_refetch=args.force_refetch
            )
            ticker_filings[t] = filings
            if (i + 1) % 50 == 0 or i == 0:
                print(f"[INFO] ({i+1}/{len(mapped)}) {t} (CIK {cik}): {len(filings)} 10-K/10-Q filings total")
        except Exception as e:
            print(f"[ERR] Failed fetching filings for {t} (CIK {cik}): {e}")

    # 5) Download relevant filings for each instance
    download_log: list = []
    ticker_has_any_data: Dict[str, bool] = {t: False for t in tickers}

    sorted_instances = sorted(instances, key=lambda x: (x[1], x[0]))
    for idx, (d, t, _) in enumerate(sorted_instances):
        cik = ticker_to_cik.get(t)
        filings = ticker_filings.get(t)
        if not cik or not filings:
            continue
        had_any = download_relevant_for_instance(sec, t, cik, d, filings, out_root, download_log)
        if had_any:
            ticker_has_any_data[t] = True
        if (idx + 1) % 100 == 0:
            print(f"[INFO] Processed {idx+1}/{len(sorted_instances)} instances...")

    # 6) Stats
    total = len(tickers)
    success = sum(1 for t in tickers if ticker_has_any_data.get(t, False))
    pct = (success / total * 100.0) if total else 0.0
    failed_tickers = sorted([t for t in tickers if not ticker_has_any_data.get(t, False)])

    print(f"\n[STATS] Tickers with any relevant 10-K/10-Q pulled: {success}/{total} ({pct:.2f}%)")
    if failed_tickers:
        no_cik_count = len(unmapped)
        has_cik_failed = [t for t in failed_tickers if t not in unmapped]
        print(f"[STATS] No CIK mapping: {no_cik_count}")
        print(f"[STATS] CIK found but no filings pulled: {len(has_cik_failed)}")
        if has_cik_failed:
            print(f"[STATS] CIK-found-but-failed tickers: {', '.join(has_cik_failed[:30])}{'...' if len(has_cik_failed) > 30 else ''}")
    else:
        print("[STATS] All tickers had at least one relevant filing pulled.")

    # Save download log
    log_path = out_root / "download_log.json"
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(download_log, f, indent=2, default=str)
    print(f"[INFO] Download log saved to {log_path}")

    # Save updated stats
    stats = {
        "total_tickers": total,
        "success_tickers": success,
        "coverage_pct": round(pct, 2),
        "unmapped_tickers": unmapped,
        "failed_with_cik": [t for t in failed_tickers if t not in unmapped],
    }
    stats_path = out_root / "coverage_stats.json"
    with stats_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"[INFO] Coverage stats saved to {stats_path}")

    # 7) Optionally download all historical filings
    if args.download_historical:
        for t in tickers:
            cik = ticker_to_cik.get(t)
            filings = ticker_filings.get(t)
            if not cik or not filings:
                continue
            download_historical_all(sec, t, cik, filings, out_root, start_date=start_dt)

    print("[DONE]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
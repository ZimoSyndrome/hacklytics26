#!/usr/bin/env python3
"""
Process SEC 10-K/10-Q filings into per-call feature vectors.

Reads HTML filings from data/reports/relevant_by_instance/{YYYYMMDD_TICKER}/
and produces data/processed/filings_call_matrix.csv aligned row-by-row with
data/processed/text_call_matrix.csv (3443 rows, same instance_id ordering).

Output columns (793 total):
  instance_id, ticker, call_date, has_10k, has_10q,
  10k_e0..10k_e383 (384 SBERT dims — 10-K MD&A, mean-pooled over chunks),
  10q_e0..10q_e383 (384 SBERT dims — 10-Q MD&A, mean-pooled over chunks),
  10k_lex_0..10k_lex_9 (10 lexical features for 10-K),
  10q_lex_0..10q_lex_9 (10 lexical features for 10-Q)

Filing date safety: each relevant_by_instance/{YYYYMMDD_TICKER}/ folder was
populated by sec_filings.py with the latest filing where filing_date < call_date.
No look-ahead bias.

Usage:
    python3 process_filings.py [--force] [--no-embed]
        [--reports-root data/reports/relevant_by_instance]
        [--matrix-ref   data/processed/text_call_matrix.csv]
        [--output       data/processed/filings_call_matrix.csv]
        [--cache-file   data/processed/filings_cache.json]
        [--model        all-MiniLM-L6-v2]
        [--batch-size   32]
        [--log-file     data/filings_log.json]
"""

import argparse
import csv
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Regex for filing filenames: TICKER_10-K_YYYY-MM-DD_docname.htm
FILING_RE = re.compile(
    r'^([A-Z0-9.\-]+)_(10-[KQ])_(\d{4}-\d{2}-\d{2})_(.+\.htm)$',
    re.IGNORECASE,
)

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
CHUNK_SIZE = 256    # words per chunk
CHUNK_OVERLAP = 32  # word overlap between adjacent chunks

# Lexical feature names — order defines column indices 0-9
LEX_FEATURE_NAMES = [
    "doc_length",               # 0 total word count in extracted section
    "n_risk_paragraphs",        # 1 paragraphs containing "risk"
    "going_concern_count",      # 2 "going concern"
    "material_weakness_count",  # 3 "material weakness"
    "restatement_count",        # 4 "restat(e|ed|ement)"
    "investigation_count",      # 5 "investigation|SEC|inquiry"
    "uncertainty_count",        # 6 "uncertain(ty|ties)"
    "litigation_count",         # 7 "litigation|lawsuit|legal proceedings"
    "negative_word_count",      # 8 "decline|loss|decrease|impair|write-?down|write-?off"
    "readability_sentences",    # 9 sentence count
]


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

def build_output_header() -> List[str]:
    """Return the 793-element list of CSV column names in canonical order."""
    cols = ["instance_id", "ticker", "call_date", "has_10k", "has_10q"]
    cols += [f"10k_e{i}" for i in range(EMBEDDING_DIM)]        # 384
    cols += [f"10q_e{i}" for i in range(EMBEDDING_DIM)]        # 384
    cols += [f"10k_lex_{i}" for i in range(len(LEX_FEATURE_NAMES))]  # 10
    cols += [f"10q_lex_{i}" for i in range(len(LEX_FEATURE_NAMES))]  # 10
    assert len(cols) == 793, f"Header length {len(cols)} != 793"
    return cols


# ---------------------------------------------------------------------------
# Instance ordering
# ---------------------------------------------------------------------------

def load_instance_order(matrix_ref_path: Path) -> List[str]:
    """Read instance_id ordering from text_call_matrix.csv.

    Returns an ordered list of 3,443 instance_id strings that defines the
    canonical row order filings_call_matrix.csv must match exactly.
    """
    ids: List[str] = []
    with matrix_ref_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ids.append(row["instance_id"])
    return ids


# ---------------------------------------------------------------------------
# Instance ID parsing
# ---------------------------------------------------------------------------

def parse_instance_id(inst_name: str) -> Tuple[str, str, str]:
    """Parse 'YYYYMMDD_TICKER' -> (instance_id, ticker, call_date_iso).

    Example: '20150225_LMAT' -> ('20150225_LMAT', 'LMAT', '2015-02-25')
    """
    parts = inst_name.split("_", 1)
    date_str = parts[0]
    ticker = parts[1] if len(parts) > 1 else ""
    call_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return inst_name, ticker, call_date


# ---------------------------------------------------------------------------
# Filing discovery
# ---------------------------------------------------------------------------

def discover_instance_filings(inst_dir: Path) -> Dict[str, Optional[Path]]:
    """Scan an instance directory for 10-K and 10-Q .htm filing files.

    Returns:
        {"10-K": Path|None, "10-Q": Path|None}

    Rules:
    - Files not matching FILING_RE (e.g. trailing '_', no .htm extension) are skipped.
    - If multiple files match the same form type, picks the one with the LATEST
      filing date string (ISO format sorts correctly as strings).
    - Empty directory -> both None.
    """
    result: Dict[str, Optional[Path]] = {"10-K": None, "10-Q": None}

    if not inst_dir.is_dir():
        return result

    candidates: Dict[str, List[Tuple[str, Path]]] = {"10-K": [], "10-Q": []}

    for f in inst_dir.iterdir():
        m = FILING_RE.match(f.name)
        if m is None:
            continue
        form = m.group(2).upper()        # "10-K" or "10-Q"
        filing_date_str = m.group(3)     # "YYYY-MM-DD"
        if form in candidates:
            candidates[form].append((filing_date_str, f))

    for form in ("10-K", "10-Q"):
        if candidates[form]:
            # Latest by filing date (ISO strings sort lexicographically)
            candidates[form].sort(key=lambda x: x[0], reverse=True)
            result[form] = candidates[form][0][1]

    return result


# ---------------------------------------------------------------------------
# HTML parsing
# ---------------------------------------------------------------------------

def strip_xbrl_header(raw_html: str) -> str:
    """Remove the EDGAR XBRL wrapper that precedes the <HTML> tag.

    EDGAR native files begin with <DOCUMENT>/<TYPE>/... before <HTML>.
    Returns everything from the first <HTML> or <html> tag onward.
    If no such tag found, returns the original string.
    """
    idx = raw_html.find("<HTML")
    if idx == -1:
        idx = raw_html.find("<html")
    if idx == -1:
        return raw_html
    return raw_html[idx:]


def extract_section_raw(raw_html: str, form_type: str) -> Tuple[int, int]:
    """Locate MD&A section boundaries in raw HTML by character offset.

    Strategy (stops at first success):
    1. Search for Item 7 (10-K) or Item 2 (10-Q) header pattern.
       Use the SECOND occurrence if >=2 found (first is usually the TOC).
    2. Fallback: 'Management...Discussion' pattern, second occurrence.
    3. Last resort: (0, len(raw_html)) — full document.

    Section end: next Item 8/9 (10-K) or Item 3 (10-Q), or PART II/III.

    Returns:
        (start_char, end_char) offsets into raw_html.
    """
    item_num = "7" if form_type == "10-K" else "2"

    # --- Section start ---
    item_pat = re.compile(
        r'[Ii]tem(?:&nbsp;|&#160;|\s)+' + item_num + r'(?:[.\s&<]|$)',
        re.IGNORECASE,
    )
    matches = list(item_pat.finditer(raw_html))

    start: Optional[int] = None
    if len(matches) >= 2:
        start = matches[1].start()
    elif len(matches) == 1:
        start = matches[0].start()

    # Fallback: Management's Discussion
    if start is None:
        mgmt_pat = re.compile(r'Management.{0,30}Discussion', re.IGNORECASE)
        mgmt_matches = list(mgmt_pat.finditer(raw_html))
        if mgmt_matches:
            idx = min(1, len(mgmt_matches) - 1)
            start = mgmt_matches[idx].start()

    # Last resort: full document
    if start is None:
        return 0, len(raw_html)

    # --- Section end ---
    if form_type == "10-K":
        next_item_pat = re.compile(
            r'[Ii]tem(?:&nbsp;|&#160;|\s)+[89](?:[.\s&<]|$)|PART\s+(?:II|III)',
            re.IGNORECASE,
        )
    else:
        next_item_pat = re.compile(
            r'[Ii]tem(?:&nbsp;|&#160;|\s)+3(?:[.\s&<]|$)|PART\s+(?:II|III)',
            re.IGNORECASE,
        )

    end_match = next_item_pat.search(raw_html, pos=start + 100)
    end = end_match.start() if end_match else len(raw_html)

    return start, end


def parse_html_to_text(html_content: str) -> str:
    """Parse HTML string to clean plain text using BeautifulSoup + lxml.

    Removes: table, thead, tbody, tr, td, th, script, style.
    Returns whitespace-normalised plain text.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, "lxml")

    for tag in soup.find_all(["table", "thead", "tbody", "tr", "td", "th",
                               "script", "style"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_and_parse_filing(filing_path: Path, form_type: str) -> Tuple[str, bool]:
    """Load an HTML filing and extract MD&A section as clean plain text.

    Args:
        filing_path: path to the .htm file
        form_type:   "10-K" or "10-Q"

    Returns:
        (text, section_found)
        text:           clean plain text of the MD&A (or full doc as fallback)
        section_found:  True if Item 7/2 or MD&A was successfully located

    Handles encoding errors, degenerate files, and parse failures gracefully.
    """
    # --- Read file with encoding fallback ---
    raw = None
    for encoding in ("utf-8", "latin-1", "cp1252"):
        try:
            with filing_path.open("r", encoding=encoding, errors="strict") as f:
                raw = f.read()
            break
        except (UnicodeDecodeError, LookupError):
            continue

    if raw is None:
        with filing_path.open("r", encoding="utf-8", errors="replace") as f:
            raw = f.read()

    if not raw.strip():
        return "", False

    # --- Locate section in raw HTML ---
    start, end = extract_section_raw(raw, form_type)
    section_found = not (start == 0 and end == len(raw))

    section_html = raw[start:end]

    # If full document is being used, strip the XBRL header first
    if start == 0:
        section_html = strip_xbrl_header(section_html)

    # Degenerate content check (e.g. SRCL directory listing HTML)
    if len(section_html.strip()) < 500:
        return "", False

    # --- Parse HTML to text ---
    try:
        text = parse_html_to_text(section_html)
    except Exception as e:
        print(f"[WARN] HTML parse failed for {filing_path.name}: {e}")
        return "", False

    # Final length check
    if len(text.split()) < 10:
        return "", False

    return text, section_found


# ---------------------------------------------------------------------------
# Chunking and embedding
# ---------------------------------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """Split text into overlapping word-level chunks.

    Always returns at least one chunk even if text is shorter than chunk_size.
    """
    words = text.split()
    if not words:
        return [""]

    step = max(1, chunk_size - overlap)
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i += step

    return chunks if chunks else [""]


def embed_text(text: str, model) -> np.ndarray:
    """Embed text via SBERT chunking + mean pooling.

    Steps:
      1. Chunk text into 256-word chunks with 32-word overlap.
      2. Encode all chunks with the SBERT model.
      3. Mean-pool chunk embeddings -> single 384-dim vector.

    Returns np.zeros(384) for empty text or on failure.
    """
    if not text or not text.strip():
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    chunks = chunk_text(text)

    try:
        embeddings = model.encode(
            chunks,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=False,
            convert_to_numpy=True,
        )
        mean_vec = embeddings.mean(axis=0)
        return mean_vec.astype(np.float32)
    except Exception as e:
        print(f"[WARN] Embedding failed: {e}")
        return np.zeros(EMBEDDING_DIM, dtype=np.float32)


# ---------------------------------------------------------------------------
# Lexical features
# ---------------------------------------------------------------------------

def extract_lexical_features(text: str) -> np.ndarray:
    """Extract 10 lexical features from plain text.

    Feature order matches LEX_FEATURE_NAMES (indices 0-9).
    Returns np.zeros(10) for empty text.
    """
    if not text or not text.strip():
        return np.zeros(len(LEX_FEATURE_NAMES), dtype=np.float64)

    words = text.split()
    doc_length = len(words)

    # Risk paragraphs — use double-newline or multi-space as paragraph boundary
    # (text is HTML-stripped so \n\n is rare; use '. ' as sentence proxy)
    paragraphs = re.split(r"\n{2,}|\. {2,}", text)
    n_risk_paragraphs = sum(
        1 for p in paragraphs if re.search(r"\brisk\b", p, re.IGNORECASE)
    )

    going_concern_count = len(re.findall(r"going concern", text, re.IGNORECASE))
    material_weakness_count = len(re.findall(r"material weakness", text, re.IGNORECASE))
    restatement_count = len(re.findall(
        r"restat(?:e|ed|ement|ements)", text, re.IGNORECASE
    ))
    investigation_count = len(re.findall(
        r"\binvestigation\b|\bSEC\b|\binquiry\b", text, re.IGNORECASE
    ))
    uncertainty_count = len(re.findall(
        r"\buncertain(?:ty|ties)?\b", text, re.IGNORECASE
    ))
    litigation_count = len(re.findall(
        r"\blitigation\b|\blawsuit\b|\blegal proceedings\b", text, re.IGNORECASE
    ))
    negative_word_count = len(re.findall(
        r"\bdecline\b|\bloss\b|\bdecrease\b|\bimpair\b|\bwrite-?down\b|\bwrite-?off\b",
        text, re.IGNORECASE,
    ))

    sentences = re.split(r"[.!?]+\s+", text.strip())
    readability_sentences = len([s for s in sentences if s.strip()])

    return np.array([
        doc_length,
        n_risk_paragraphs,
        going_concern_count,
        material_weakness_count,
        restatement_count,
        investigation_count,
        uncertainty_count,
        litigation_count,
        negative_word_count,
        readability_sentences,
    ], dtype=np.float64)


# ---------------------------------------------------------------------------
# Per-instance processing
# ---------------------------------------------------------------------------

def process_one_instance(
    inst_name: str,
    reports_root: Path,
    model,          # SentenceTransformer | None
) -> dict:
    """Process one MAEC instance: find its filings, extract features.

    Returns a dict with keys:
        instance_id, ticker, call_date,
        has_10k, has_10q,
        emb_10k (384,), emb_10q (384,),
        lex_10k (10,), lex_10q (10,),
        status, detail
    """
    inst_id, ticker, call_date = parse_instance_id(inst_name)

    zero_emb = np.zeros(EMBEDDING_DIM, dtype=np.float32)
    zero_lex = np.zeros(len(LEX_FEATURE_NAMES), dtype=np.float64)

    result: dict = {
        "instance_id": inst_id,
        "ticker": ticker,
        "call_date": call_date,
        "has_10k": 0,
        "has_10q": 0,
        "emb_10k": zero_emb.copy(),
        "emb_10q": zero_emb.copy(),
        "lex_10k": zero_lex.copy(),
        "lex_10q": zero_lex.copy(),
        "status": "ok",
        "detail": "",
    }

    inst_dir = reports_root / inst_name

    if not inst_dir.is_dir():
        result["status"] = "no_folder"
        result["detail"] = "no relevant_by_instance folder"
        return result

    filings = discover_instance_filings(inst_dir)
    details: List[str] = []

    for form_type in ("10-K", "10-Q"):
        filing_path = filings.get(form_type)
        key = "10k" if form_type == "10-K" else "10q"

        if filing_path is None:
            details.append(f"no {form_type} file")
            continue

        try:
            text, section_found = load_and_parse_filing(filing_path, form_type)
        except Exception as e:
            details.append(f"{form_type} parse error: {str(e)[:100]}")
            continue

        if not text:
            details.append(f"{form_type} empty text after parse")
            continue

        result[f"has_{key}"] = 1

        if model is not None:
            result[f"emb_{key}"] = embed_text(text, model)

        result[f"lex_{key}"] = extract_lexical_features(text)

        n_words = len(text.split())
        n_chunks = len(chunk_text(text))
        details.append(
            f"{form_type}: ok (section={section_found}, words={n_words}, chunks={n_chunks})"
        )

    has_10k = result["has_10k"]
    has_10q = result["has_10q"]
    if has_10k == 0 and has_10q == 0:
        if not inst_dir.is_dir():
            result["status"] = "no_folder"
        else:
            result["status"] = "no_filings"
    elif has_10k == 0 or has_10q == 0:
        result["status"] = "partial"

    result["detail"] = "; ".join(details) if details else ""
    return result


# ---------------------------------------------------------------------------
# Row assembly
# ---------------------------------------------------------------------------

def assemble_row(proc_result: dict) -> List:
    """Flatten a process_one_instance result dict into a 793-element CSV row.

    Column order: instance_id, ticker, call_date, has_10k, has_10q,
                  10k embeddings (384), 10q embeddings (384),
                  10k lexical (10), 10q lexical (10).
    """
    row: List = [
        proc_result["instance_id"],
        proc_result["ticker"],
        proc_result["call_date"],
        proc_result["has_10k"],
        proc_result["has_10q"],
    ]

    for v in proc_result["emb_10k"]:
        row.append(f"{v:.8f}")
    for v in proc_result["emb_10q"]:
        row.append(f"{v:.8f}")
    for v in proc_result["lex_10k"]:
        row.append(f"{v:.2f}")
    for v in proc_result["lex_10q"]:
        row.append(f"{v:.2f}")

    assert len(row) == 793, f"Row length {len(row)} != 793"
    return row


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

def load_cache(cache_path: Path) -> Dict[str, dict]:
    """Load previously computed per-instance results from JSON cache.

    Converts stored lists back to numpy arrays.
    """
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        print(f"[WARN] Cache file unreadable, starting fresh: {cache_path}")
        return {}

    for rec in data.values():
        for key in ("emb_10k", "emb_10q"):
            rec[key] = np.array(rec[key], dtype=np.float32)
        for key in ("lex_10k", "lex_10q"):
            rec[key] = np.array(rec[key], dtype=np.float64)
    return data


def save_cache(cache_path: Path, cache: Dict[str, dict]) -> None:
    """Save per-instance results to JSON cache (numpy -> lists)."""
    serializable: dict = {}
    for inst_id, rec in cache.items():
        s = dict(rec)
        for key in ("emb_10k", "emb_10q", "lex_10k", "lex_10q"):
            s[key] = rec[key].tolist()
        serializable[inst_id] = s

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(serializable, f)
    tmp_path.replace(cache_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Process SEC 10-K/10-Q filings into per-call feature vectors."
    )
    ap.add_argument(
        "--reports-root",
        default="data/reports/relevant_by_instance",
        help="Root dir containing YYYYMMDD_TICKER subdirs with .htm filings.",
    )
    ap.add_argument(
        "--matrix-ref",
        default="data/processed/text_call_matrix.csv",
        help="Reference CSV whose instance_id column defines canonical row order.",
    )
    ap.add_argument(
        "--output",
        default="data/processed/filings_call_matrix.csv",
        help="Output CSV path.",
    )
    ap.add_argument(
        "--cache-file",
        default="data/processed/filings_cache.json",
        help="JSON cache for resumability.",
    )
    ap.add_argument(
        "--model",
        default=SBERT_MODEL_NAME,
        help="SBERT model name.",
    )
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="SBERT encoding batch size.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Ignore cache; reprocess all instances.",
    )
    ap.add_argument(
        "--no-embed",
        action="store_true",
        help="Skip SBERT embedding (fill with zeros). Fast path for lexical-only runs.",
    )
    ap.add_argument(
        "--log-file",
        default="data/filings_log.json",
        help="Processing log path.",
    )
    args = ap.parse_args()

    reports_root = Path(args.reports_root)
    matrix_ref_path = Path(args.matrix_ref)
    output_path = Path(args.output)
    cache_path = Path(args.cache_file)
    log_path = Path(args.log_file)

    # --- Validate inputs ---
    if not reports_root.is_dir():
        print(f"[ERR] Reports root not found: {reports_root}")
        return 1
    if not matrix_ref_path.exists():
        print(f"[ERR] Reference matrix not found: {matrix_ref_path}")
        return 1

    # --- Load canonical instance order ---
    print(f"[INFO] Loading instance order from {matrix_ref_path}")
    instance_order = load_instance_order(matrix_ref_path)
    print(f"[INFO] Canonical row count: {len(instance_order)}")

    # --- Load SBERT model ---
    model = None
    if not args.no_embed:
        print(f"[INFO] Loading SBERT model: {args.model}")
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(args.model)
        actual_dim = model.get_sentence_embedding_dimension()
        print(f"[INFO] Embedding dimension: {actual_dim}")
        if actual_dim != EMBEDDING_DIM:
            print(f"[WARN] Expected {EMBEDDING_DIM} dims, got {actual_dim}. Output columns will be wrong.")
    else:
        print("[INFO] --no-embed: skipping SBERT, embedding columns will be all zeros.")

    # --- Load cache ---
    cache: Dict[str, dict] = {}
    if not args.force:
        cache = load_cache(cache_path)
        print(f"[INFO] Loaded {len(cache)} cached instance results.")
    else:
        print("[INFO] --force: ignoring cache.")

    # --- Build output header ---
    header = build_output_header()

    # --- Processing stats ---
    stats: Dict[str, int] = {
        "ok": 0, "partial": 0, "no_folder": 0, "no_filings": 0,
        "error": 0, "cached": 0,
    }

    log: dict = {
        "started_at": datetime.now().isoformat(),
        "model": args.model if not args.no_embed else "none (--no-embed)",
        "force": args.force,
        "no_embed": args.no_embed,
        "total_instances": len(instance_order),
        "results": [],
    }

    t_start = time.time()
    rows: List[List] = []

    for idx, inst_name in enumerate(instance_order):

        # Cache hit?
        if inst_name in cache and not args.force:
            proc_result = cache[inst_name]
            stats["cached"] += 1
        else:
            try:
                proc_result = process_one_instance(inst_name, reports_root, model)
            except Exception as e:
                # Hard failure — fill with zeros, keep going
                inst_id, ticker, call_date = parse_instance_id(inst_name)
                proc_result = {
                    "instance_id": inst_id,
                    "ticker": ticker,
                    "call_date": call_date,
                    "has_10k": 0,
                    "has_10q": 0,
                    "emb_10k": np.zeros(EMBEDDING_DIM, dtype=np.float32),
                    "emb_10q": np.zeros(EMBEDDING_DIM, dtype=np.float32),
                    "lex_10k": np.zeros(len(LEX_FEATURE_NAMES), dtype=np.float64),
                    "lex_10q": np.zeros(len(LEX_FEATURE_NAMES), dtype=np.float64),
                    "status": "error",
                    "detail": str(e)[:300],
                }
                print(f"[ERR] {inst_name}: {e}")

            cache[inst_name] = proc_result

        status = proc_result.get("status", "ok")
        stats[status] = stats.get(status, 0) + 1

        row = assemble_row(proc_result)
        rows.append(row)

        log["results"].append({
            "instance": inst_name,
            "status": status,
            "has_10k": proc_result["has_10k"],
            "has_10q": proc_result["has_10q"],
            "detail": proc_result.get("detail", ""),
        })

        # Progress + incremental cache save
        if (idx + 1) % 200 == 0:
            elapsed = time.time() - t_start
            done = idx + 1
            rate = done / elapsed
            eta = (len(instance_order) - done) / rate
            print(
                f"[INFO] ({done}/{len(instance_order)}) "
                f"ok={stats['ok']} partial={stats['partial']} "
                f"no_folder={stats['no_folder']} cached={stats['cached']} | "
                f"elapsed={elapsed:.0f}s ETA={eta:.0f}s"
            )
            save_cache(cache_path, cache)

    # --- Write output CSV ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(row)

    print(f"[INFO] Written {len(rows)} rows × {len(header)} cols to {output_path}")

    # --- Final cache save ---
    save_cache(cache_path, cache)

    # --- Write log ---
    elapsed_total = time.time() - t_start
    log["completed_at"] = datetime.now().isoformat()
    log["elapsed_seconds"] = round(elapsed_total, 1)
    log["summary"] = dict(stats)
    log["output_rows"] = len(rows)
    log["output_cols"] = len(header)

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, default=str)

    # --- Summary ---
    print(f"\n{'='*60}")
    print("FILINGS PROCESSING COMPLETE")
    print(f"{'='*60}")
    print(f"  Instances:          {len(instance_order)}")
    print(f"  From cache:         {stats['cached']}")
    print(f"  OK (both filings):  {stats.get('ok', 0)}")
    print(f"  Partial (1 filing): {stats.get('partial', 0)}")
    print(f"  No folder:          {stats.get('no_folder', 0)}")
    print(f"  No filings (empty): {stats.get('no_filings', 0)}")
    print(f"  Errors:             {stats.get('error', 0)}")
    print(f"  Output shape:       {len(rows)} × {len(header)}")
    print(f"  Output:             {output_path}")
    print(f"  Log:                {log_path}")
    print(f"  Elapsed:            {elapsed_total:.1f}s")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

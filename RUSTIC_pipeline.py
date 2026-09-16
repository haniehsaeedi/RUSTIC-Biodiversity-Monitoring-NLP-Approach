"""
pipeline.py — End-to-end RUSTIC document discovery pipeline.

Steps
-----
1. Fetch document metadata page-by-page from OpenAlex (using the query in config.py).
2. For each document, attempt to download the full text (PDF preferred; HTML fallback).
3. Score the text with the pre-trained RUSTIC classifier + definition scorer.
4. If the combined score exceeds COMBO_THRESHOLD, append the document to the output Excel.
5. Run until OpenAlex is exhausted (TARGET_RUSTIC_MATCHES = None) or the target is reached.

Resume behaviour
----------------
- Every OpenAlex ID that has been processed (matched or not) is saved to SEEN_IDS_PATH.
- On restart the file is loaded and those IDs are skipped immediately, so no document
  is downloaded or scored twice across multiple runs.
- New matches are APPENDED to the existing OUTPUT_EXCEL, never overwriting previous results.

Usage
-----
    python pipeline.py
"""

import math
import os
import re
import shutil
import tempfile
import time

import joblib
import nltk
import openpyxl
import pandas as pd
import requests
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from pyalex import Works, config as pyalex_config
from urllib.parse import urljoin

import config
from model_utils import (
    TfidfCosineScorer,          # noqa: F401  — must be imported so joblib pickle finds it
    bm25_best_window_score,
    cleanup_text,
    download_to_temp,
    extract_text_from_pdf_path,
    simple_tokenize,
)
from stats_tracker import StatsTracker

# ── Configure pyalex ──────────────────────────────────────────────────────────
pyalex_config.email = config.OPENALEX_EMAIL

# API key must be set on pyalex.config — pyalex's .get() does not accept it
# as a keyword argument; it reads it from the global config object instead.
_api_key = config.OPENALEX_API_KEY.strip()
if _api_key and _api_key != "YOUR_API_KEY_HERE":
    pyalex_config.api_key = _api_key

# ── NLTK resources ────────────────────────────────────────────────────────────
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEMMER   = PorterStemmer()


# ─────────────────────────────────────────────────────────────────────────────
# Seen-ID persistence  (enables safe stop/resume across days)
# ─────────────────────────────────────────────────────────────────────────────

def load_seen_ids(path: str) -> set[str]:
    """Load previously processed OpenAlex IDs from disk."""
    if not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        ids = {line.strip() for line in f if line.strip()}
    print(f"   Resuming: {len(ids):,} IDs already processed — will skip them.")
    return ids


def save_seen_id(path: str, openalex_id: str) -> None:
    """Append a single processed ID to the seen-IDs file immediately."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(openalex_id + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction from downloaded file
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return cleanup_text(extract_text_from_pdf_path(path))
    if path.lower().endswith((".txt", ".html")):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return cleanup_text(f.read())
        except Exception:
            return ""
    try:
        return cleanup_text(extract_text_from_pdf_path(path))
    except Exception:
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Boilerplate detection  (skip publisher landing pages, cookie walls, etc.)
# ─────────────────────────────────────────────────────────────────────────────

_BOILERPLATE_SIGNALS = [
    "Our AI tools perform quality checks",
    "This article is behind a paywall",
    "Please enable JavaScript",
    "Access provided by",
    "Sign in to view full text",
    "Purchase this article",
]


def is_boilerplate(text: str) -> bool:
    return any(signal in text for signal in _BOILERPLATE_SIGNALS)


# ─────────────────────────────────────────────────────────────────────────────
# RUSTIC scoring  (identical logic to notebook Cell 2)
# ─────────────────────────────────────────────────────────────────────────────

def stem_filter(tokens: list[str]) -> list[str]:
    return [STEMMER.stem(t) for t in tokens
            if t not in STOPWORDS and len(t) > 2]


def score_document(text: str, pipeline, threshold: float, scorer,
                   rustic_definitions: dict) -> dict:
    """Score one document. Returns the same fields as notebook Cell 2."""
    if not text:
        return {
            "Model_Prob":            0.0,
            "Best_Def_Score":        0.0,
            "Best_Definition":       "",
            "Best_Matched_Text":     "",
            "Initial_Keywords_Hit":  "",
            **{f"Score_{k}": 0.0 for k in rustic_definitions},
        }

    try:
        prob = float(pipeline.predict_proba([text])[:, 1][0])
    except Exception:
        prob = 0.0

    best_def, best_score, best_span, keywords_hit = None, -1.0, "", ""
    per_def_scores: dict[str, float] = {}

    for def_name, def_text in rustic_definitions.items():
        try:
            tfidf_sim, tfidf_span = scorer.best_window(def_text, text)
        except Exception:
            tfidf_sim, tfidf_span = 0.0, ""

        bm25_score, bm25_span = bm25_best_window_score(def_text, text)
        bm25_norm  = 1 - math.exp(-bm25_score)
        def_score  = 0.7 * tfidf_sim + 0.3 * bm25_norm
        span       = tfidf_span if tfidf_sim >= bm25_norm else bm25_span

        kw_tokens    = set(stem_filter(simple_tokenize(span)))
        def_keywords = sorted(
            set(stem_filter(simple_tokenize(def_text))),
            key=lambda x: (-len(x), x),
        )[:12]
        hits = [k for k in def_keywords if k in kw_tokens]

        per_def_scores[def_name] = float(def_score)
        if def_score > best_score:
            best_score   = def_score
            best_def     = def_name
            best_span    = span
            keywords_hit = ", ".join(hits)

    return {
        "Model_Prob":            prob,
        "Best_Def_Score":        float(best_score),
        "Best_Definition":       best_def or "",
        "Best_Matched_Text":     best_span,
        "Initial_Keywords_Hit":  keywords_hit,
        **{f"Score_{k}": per_def_scores[k] for k in rustic_definitions},
    }


def combined_score(model_prob: float, best_def_score: float) -> float:
    return model_prob * config.MODEL_WEIGHT + best_def_score * (1 - config.MODEL_WEIGHT)


# ─────────────────────────────────────────────────────────────────────────────
# OpenAlex pagination
# ─────────────────────────────────────────────────────────────────────────────

def openalex_iterator(page_size: int = 100, result_holder: dict | None = None):
    """
    Fetch OpenAlex results using pyalex's built-in .paginate() cursor method.

    Key facts (as of Feb 2026):
    - API key is REQUIRED. Without one: only 100 credits/day (approx 100 pages).
    - With free key: 100,000 credits/day (approx 100,000 filter pages).
    - List+filter requests: 1 credit each.
    - Max per_page: 100.
    - pyalex.config handles retries, backoff, key — set once, applies everywhere.

    result_holder: optional dict populated with {"total_count": int} from first page.
    """
    import pyalex as _pyalex

    # ── Configure pyalex once — applies to all requests automatically ─────────
    api_key = config.OPENALEX_API_KEY.strip()
    if api_key and api_key != "YOUR_API_KEY_HERE":
        _pyalex.config.api_key   = api_key
        _pyalex.config.email     = config.OPENALEX_EMAIL
        print(f"  [OpenAlex] API key active ({api_key[:8]}...)")
    else:
        print("  WARNING: No API key set — only 100 credits/day (approx 100 pages).")
        print("           Get a free key at https://openalex.org/settings/api")
        print("           Then set OPENALEX_API_KEY in config.py")

    # Built-in retry — handles 429 and 503 automatically without manual loops
    _pyalex.config.max_retries          = 10
    _pyalex.config.retry_backoff_factor = 0.5
    _pyalex.config.retry_http_codes     = [429, 500, 503]

    # ── Build query ───────────────────────────────────────────────────────────
    query = Works().filter(**config.OPENALEX_FILTER).select(
        "id,doi,title,publication_year,primary_location"
    )

    # Get total count first so we can report progress
    try:
        total = query.count()
        print(f"  [OpenAlex] Total results for this query: {total:,}")
        if result_holder is not None:
            result_holder["total_count"] = total
    except Exception as e:
        print(f"  [OpenAlex] Could not fetch total count: {e}")

    # ── Paginate — pyalex handles cursor advancement automatically ────────────
    page_num = 0
    try:
        for page in query.paginate(per_page=page_size, n_max=None):
            page_num += 1
            if page_num % 10 == 0:
                print(f"  [OpenAlex] Page {page_num} fetched ({len(page)} results) ...")
            for work in page:
                yield work
    except Exception as e:
        print(f"\n  [OpenAlex] Pagination stopped at page {page_num}: {e}")
        print("  If this is a 429, your daily budget is exhausted.")
        print("  Check https://openalex.org/settings/usage — resets at midnight UTC.")





# ─────────────────────────────────────────────────────────────────────────────
# Excel output  — APPEND mode so multiple runs accumulate results
# ─────────────────────────────────────────────────────────────────────────────

_OUTPUT_COLS = [
    "Title",
    "Link",
    "DOI",
    "Publication Year",
    "Source Query",
    "RUSTIC definitions context match score",
    "Best_Matched_Text",
    "_Pred_Label_by_match_score",
    "RUSTIC_Match",
    "Best_Definition",
    "Initial_Keywords_Hit",
    "Model_Prob",
    "Best_Def_Score",
]


def _sanitize_cell(value, max_len: int = 32_000):
    """
    Make a value safe for openpyxl / lxml XML writing.
    - Converts to string if needed
    - Removes ALL XML-illegal characters including NULL bytes (\x00)
    - Truncates to Excel cell limit (32,767 chars)
    """
    if value is None:
        return ""
    if not isinstance(value, str):
        return value   # numbers, bools, dates — openpyxl handles these natively
    # Strip NULL bytes and all other XML-illegal control chars
    # XML 1.0 allows only: \x09 (tab), \x0A (LF), \x0D (CR), \x20-\uD7FF, \uE000+
    value = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", value)
    # Also remove any remaining NULL bytes that may survive the above (belt and braces)
    value = value.replace("\x00", "")
    return value[:max_len]


def append_results_to_excel(records: list[dict], output_path: str) -> None:
    """
    Append new match records to the Excel file.
    - If the file does not exist yet: create it with headers.
    - If it already exists: open and append rows — never overwrite existing data.
    """
    if not records:
        return

    df_new = pd.DataFrame(records)
    # Apply sanitizer to every column — mixed-type columns (object dtype containing
    # strings) may not be caught by select_dtypes alone, so sanitize all values.
    for col in df_new.columns:
        df_new[col] = df_new[col].apply(_sanitize_cell)

    def_score_cols = sorted([c for c in df_new.columns if c.startswith("Score_")])
    out_cols       = [c for c in _OUTPUT_COLS if c in df_new.columns] + def_score_cols

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Guard against corrupted/empty output file from a previous crashed run
    file_is_valid = False
    if os.path.exists(output_path):
        try:
            import zipfile
            with zipfile.ZipFile(output_path, 'r'):
                file_is_valid = True
        except Exception:
            print(f"\n  ⚠️  Output file appears corrupted — recreating: {output_path}")
            os.remove(output_path)

    if not file_is_valid:
        # First write or recreate after corruption — create file with headers
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            df_new[out_cols].to_excel(writer, sheet_name="Predictions", index=False)
        print(f"\n  💾  Created output file with {len(records)} matches → {output_path}")

    else:
        # Subsequent writes — open existing workbook and append rows only
        wb   = openpyxl.load_workbook(output_path)
        ws   = wb["Predictions"]

        # Align columns to whatever order the file already has
        existing_headers = [cell.value for cell in ws[1]]
        # Add any new columns (e.g. new Score_ columns) at the end
        for col in out_cols:
            if col not in existing_headers:
                existing_headers.append(col)

        for _, row in df_new[out_cols].iterrows():
            ws.append([_sanitize_cell(row.get(col, "")) for col in existing_headers])

        wb.save(output_path)
        print(f"\n  💾  Appended {len(records)} new matches → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    run_label = "unlimited" if config.TARGET_RUSTIC_MATCHES is None \
                else str(config.TARGET_RUSTIC_MATCHES)
    print("=" * 60)
    print("  RUSTIC OpenAlex Discovery Pipeline")
    print(f"  Target matches : {run_label}")
    print(f"  Resume file    : {config.SEEN_IDS_PATH}")
    print(f"  Output         : {config.OUTPUT_EXCEL}")
    print(f"  Stats          : {config.STATS_JSON_PATH}")
    print(f"  Run history    : {config.STATS_RUNS_CSV}")
    print("=" * 60)

    # ── Load model ────────────────────────────────────────────────────────────
    if not os.path.exists(config.MODEL_PATH) or not os.path.exists(config.SCORER_PATH):
        raise FileNotFoundError(
            "Model files not found. Run train.py first.\n"
            f"  Expected: {config.MODEL_PATH}\n"
            f"            {config.SCORER_PATH}"
        )
    mp        = joblib.load(config.MODEL_PATH)
    pipeline  = mp["pipeline"]
    threshold = mp.get("threshold", 0.5)

    sp          = joblib.load(config.SCORER_PATH)
    scorer      = sp["scorer"]
    rustic_defs = sp["definitions"]
    print(f"\n✅  Model loaded  (threshold={threshold:.2f})")

    # ── Load seen IDs from previous runs ─────────────────────────────────────
    seen_ids: set[str] = load_seen_ids(config.SEEN_IDS_PATH)

    # ── Initialise stats tracker ──────────────────────────────────────────────
    stats = StatsTracker(config.STATS_JSON_PATH, config.STATS_RUNS_CSV)
    openalex_result_holder: dict = {}

    # ── Temp directory for downloads ──────────────────────────────────────────
    tmpdir = tempfile.mkdtemp(prefix="rustic_pipeline_")

    # Accumulate new matches in memory; flush to Excel every CHECKPOINT_EVERY
    pending_matches: list[dict] = []
    total_matches_this_run = 0
    processed_this_run     = 0

    try:
        print(f"\n🔍  Querying OpenAlex:")
        print(f"    Filter    : {config.OPENALEX_FILTER}")
        key_display = config.OPENALEX_API_KEY[:8] + "..." \
            if config.OPENALEX_API_KEY and config.OPENALEX_API_KEY != "YOUR_API_KEY_HERE" \
            else "NOT SET (anonymous)"
        print(f"    API key   : {key_display}\n")
        print("    Waiting 30 s before first request ...")
        time.sleep(30)

        for work in openalex_iterator(
            page_size=config.OPENALEX_PAGE_SIZE,
            result_holder=openalex_result_holder,
        ):

            work_id = work.get("id", "")

            # ── Skip already-processed documents ──────────────────────────────
            if work_id in seen_ids:
                continue

            # Mark as seen immediately — even if download fails, don't retry
            seen_ids.add(work_id)
            save_seen_id(config.SEEN_IDS_PATH, work_id)

            title = work.get("title") or ""
            doi   = work.get("doi") or ""
            year  = work.get("publication_year") or ""

            location      = work.get("primary_location") or {}
            pdf_url       = location.get("pdf_url")
            landing_url   = location.get("landing_page_url")
            resource_link = pdf_url or landing_url or ""

            if not resource_link:
                stats.record("no_resource_link")
                continue

            processed_this_run += 1
            print(
                f"[processed: {processed_this_run:>6} | "
                f"matches this run: {total_matches_this_run:>4} | "
                f"total seen: {len(seen_ids):>7}]  {title[:45]!r}",
                end="\r",
            )

            # ── Download ──────────────────────────────────────────────────────
            tmp_path = download_to_temp(
                resource_link, tmpdir,
                timeout=config.HTTP_TIMEOUT,
                user_agent=config.HTTP_USER_AGENT,
            )
            download_ok = bool(tmp_path)

            if not tmp_path:
                try:
                    r = requests.get(
                        resource_link, timeout=config.HTTP_TIMEOUT,
                        headers={"User-Agent": config.HTTP_USER_AGENT},
                    )
                    ct = r.headers.get("Content-Type", "")
                    if "text/html" in ct:
                        soup = BeautifulSoup(r.text, "lxml")
                        text = cleanup_text(soup.get_text(separator=" "))
                        download_ok = True
                    else:
                        text = ""
                except Exception:
                    text = ""
            else:
                text = extract_text_from_file(tmp_path)
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

            if not download_ok:
                stats.record("download_failed")
                continue

            if text and is_boilerplate(text):
                stats.record("boilerplate_detected")
                continue

            if not text:
                stats.record("empty_text")
                continue

            # ── Score ─────────────────────────────────────────────────────────
            scores = score_document(text, pipeline, threshold, scorer, rustic_defs)
            combo  = combined_score(scores["Model_Prob"], scores["Best_Def_Score"])

            if combo < config.COMBO_THRESHOLD:
                stats.record("below_threshold")
                continue

            # ── Match ─────────────────────────────────────────────────────────
            stats.record("matched")
            record = {
                "Title":            title,
                "Link":             resource_link,
                "DOI":              doi,
                "Publication Year": year,
                "Source Query":     "Evidence type 1 (TA)",
                "RUSTIC definitions context match score": round(combo, 4),
                "_Pred_Label_by_match_score": 1,
                "RUSTIC_Match":     "Yes",
                **scores,
            }
            pending_matches.append(record)
            total_matches_this_run += 1

            print(
                f"\n  ✅  Match #{total_matches_this_run:<4} "
                f"score={combo:.3f}  {title[:60]!r}"
            )

            # ── Checkpoint: flush pending matches to Excel ────────────────────
            if len(pending_matches) >= config.CHECKPOINT_EVERY:
                append_results_to_excel(pending_matches, config.OUTPUT_EXCEL)
                pending_matches = []

            # ── Optional hard stop ────────────────────────────────────────────
            if (config.TARGET_RUSTIC_MATCHES is not None
                    and total_matches_this_run >= config.TARGET_RUSTIC_MATCHES):
                print(f"\n🎯  Reached target of {config.TARGET_RUSTIC_MATCHES} matches.")
                break

            time.sleep(config.SLEEP_BETWEEN_REQUESTS)

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── Flush any remaining pending matches ───────────────────────────────────
    if pending_matches:
        append_results_to_excel(pending_matches, config.OUTPUT_EXCEL)

    # ── Persist and print full statistics ─────────────────────────────────────
    stats.finalize(
        openalex_total_results=openalex_result_holder.get("total_count"),
        seen_ids_total=len(seen_ids),
    )

    print(f"  Output         : {os.path.abspath(config.OUTPUT_EXCEL)}")
    print(f"  Resume file    : {os.path.abspath(config.SEEN_IDS_PATH)}")
    print(f"  Stats file     : {os.path.abspath(config.STATS_JSON_PATH)}")
    print(f"  Run history    : {os.path.abspath(config.STATS_RUNS_CSV)}")


if __name__ == "__main__":
    main()
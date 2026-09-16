"""
config.py — Central configuration for the RUSTIC OpenAlex Pipeline.

Edit ONLY this file to change paths, thresholds, query, or target counts.
All other modules import from here.
"""

# ── OpenAlex authentication ───────────────────────────────────────────────────
OPENALEX_EMAIL   = "vijaya.arun-nawale@senckenberg.de"

# Free API key from https://openalex.org/settings/api
# With key: $1/day free = 10,000 filter pages or 1,000 search pages.
# Without key: $0.10/day = 1/10th of the above.
# Set via pyalex.config.api_key (not as a URL param).
OPENALEX_API_KEY = "Ste5SsRrzRfTxoa0cabcoH"   # ← paste your key here

# ── OpenAlex query strategy ───────────────────────────────────────────────────
# COST COMPARISON (per 1,000 API calls):
#   .filter(title_and_abstract.search=...)  →  $0.10  (filter tier — CHEAP)
#   .search(...)                            →  $1.00  (search tier — 10× costlier)
#
# We use the filter tier: title_and_abstract.search is a FILTER field,
# not the global search endpoint, so it costs $0.10/1000 calls.
# At 100 results/page, 94k results = ~940 pages = ~$0.09 total.
#
# pyalex filter syntax for title+abstract keyword matching:
OPENALEX_FILTER = {
    "title_and_abstract.search": (
        "(monitoring) AND (species OR biodiversity OR ecosystem OR \"ecosystem service\" or \"Nature contributions to people\" OR NCP)  AND (data)"
    ),
    "is_oa": "true",
}

# Max results per page — OpenAlex hard cap is 100 (not 200).
OPENALEX_PAGE_SIZE = 100

# Maximum RUSTIC matches to collect before stopping.
# Set to None to run until OpenAlex is fully exhausted.
TARGET_RUSTIC_MATCHES = None

# ── ML model inputs (must exist before running) ───────────────────────────────
POSITIVE_PDF_DIR  = "Positive_RUSTIC_docs"   # folder of verified RUSTIC PDFs
NEGATIVE_EXCEL    = "Negative_RUSTIC.xlsx"   # Excel with a 'link' column
NEG_LINK_COL      = "link"

# ── Saved model artefacts ─────────────────────────────────────────────────────
MODEL_DIR         = "models"
MODEL_PATH        = f"{MODEL_DIR}/rustic_pipeline.joblib"
SCORER_PATH       = f"{MODEL_DIR}/rustic_scorer.joblib"

# ── Output ────────────────────────────────────────────────────────────────────
OUTPUT_DIR        = "output"
OUTPUT_EXCEL      = f"{OUTPUT_DIR}/RUSTIC_matches.xlsx"

# Tracks every OpenAlex ID processed (matched or not) so re-runs skip them.
# Delete this file only if you want to restart from scratch.
SEEN_IDS_PATH     = f"{OUTPUT_DIR}/seen_ids.txt"

# Run statistics — cumulative JSON + per-run CSV history.
# Safe to delete if you want stats to reset; has no effect on pipeline results.
STATS_JSON_PATH   = f"{OUTPUT_DIR}/pipeline_stats.json"
STATS_RUNS_CSV    = f"{OUTPUT_DIR}/pipeline_runs.csv"

# ── Download / HTTP settings ──────────────────────────────────────────────────
HTTP_TIMEOUT      = 20
HTTP_USER_AGENT   = "Senckenberg RUSTIC Pipeline (vijaya.arun-nawale@senckenberg.de)"
SLEEP_BETWEEN_REQUESTS = 0.3

# ── RUSTIC scoring thresholds ─────────────────────────────────────────────────
MODEL_WEIGHT      = 0.6
COMBO_THRESHOLD   = 0.40

# How often (number of new matches) to checkpoint-save the Excel
CHECKPOINT_EVERY  = 25

# ── RUSTIC definitions (used by scorer; do not rename keys) ───────────────────
RUSTIC_DEFINITIONS = {
    "Representative": (
        "sampling must cover the full population range (species, ecosystems, regions, time) "
        "avoid bias with a robust/random design. use weighting to reduce errors. "
        "capture variation in target variables"
    ),
    "Uncertainty": (
        "always estimate, report, and communicate uncertainty "
        "sources: sampling bias, measurement error, and model structure "
        "use confidence intervals, validation, and quality control "
        "transparency improves decisions"
    ),
    "Scalability": (
        "monitoring should work across spatial, temporal, biological, and social scales "
        "allow aggregation/disaggregation "
        "maintain data integrity "
        "states/trends may vary with scale"
    ),
    "Timeliness": (
        "data must be available fast for decision-making "
        "delays reduce usefulness "
        "regular updates enable early warnings "
        "frequency should match the pace of change "
        "start monitoring before stressors appear"
    ),
    "Interpretability": (
        "indicators must be easy to understand, linked to policy/management needs "
        "define success/failure clearly "
        "transparent workflows build trust "
        "document inputs, methods, and uncertainties "
        "use cases should be clear"
    ),
    "Comparability": (
        "enable comparison over time, space, and systems "
        "use standardised, validated methods "
        "harmonise programs and technologies through cross-validation "
        "stable procedures ensure reliable trend detection"
    ),
}

# ── Random seed ───────────────────────────────────────────────────────────────
RANDOM_STATE = 42

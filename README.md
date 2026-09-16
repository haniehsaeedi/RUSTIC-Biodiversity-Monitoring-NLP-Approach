# RUSTIC OpenAlex Discovery Pipeline

Automated pipeline that searches OpenAlex for open-access scientific documents,
downloads full text, and classifies them against the six **RUSTIC** criteria
(Representative, Uncertainty, Scalability, Timeliness, Interpretability, Comparability).

---

## File Structure

```
rustic_pipeline/
├── config.py       ← All settings live here — edit this file only
├── train.py        ← Step 1: train the classifier (run once)
├── pipeline.py     ← Step 2: run the live discovery pipeline
└── README.md
```

You also need these **input files in the same directory**:

| File / Folder | Description |
|---|---|
| `Positive_RUSTIC_docs/` | Folder of PDF files confirmed as RUSTIC-relevant (≥50 recommended) |
| `Negative_RUSTIC.xlsx` | Excel with a column named `link` containing confirmed non-RUSTIC URLs |

---

## Installation

```bash
pip install tqdm pandas requests pdfminer.six scikit-learn rank-bm25 \
            nltk joblib beautifulsoup4 lxml openpyxl pyalex
```

---

## How to Run

### Step 1 — Train the classifier (once)

```bash
python train.py
```

Produces:
- `models/rustic_pipeline.joblib`
- `models/rustic_scorer.joblib`

### Step 2 — Run the pipeline

```bash
python pipeline.py
```

The pipeline will:
1. Query OpenAlex using the search query in `config.py`
2. Download and extract text from each result (skips inaccessible documents quickly)
3. Score each document with the RUSTIC classifier
4. Save matches to `output/RUSTIC_matches.xlsx` (checkpoint saved every 25 matches)
5. Stop automatically when `TARGET_RUSTIC_MATCHES` (default 400) have been collected

---

## Configuration (`config.py`)

| Setting | Default | Description |
|---|---|---|
| `OPENALEX_QUERY` | *(biodiversity monitoring query)* | Boolean search query sent to OpenAlex |
| `TARGET_RUSTIC_MATCHES` | `400` | Stop pipeline after this many matches |
| `COMBO_THRESHOLD` | `0.40` | Minimum combined score to count as a match |
| `HTTP_TIMEOUT` | `20` | Seconds before skipping a slow download |
| `SLEEP_BETWEEN_REQUESTS` | `0.3` | Politeness delay between downloads |

---

## Output Excel (`output/RUSTIC_matches.xlsx`)

| Column | Description |
|---|---|
| `Title` | Document title from OpenAlex |
| `Link` | Direct PDF or landing page URL |
| `DOI` | Digital Object Identifier |
| `Publication Year` | Year of publication |
| `RUSTIC definitions context match score` | Combined relevance score (0–1) |
| `Best_Matched_Text` | Passage from the document most relevant to RUSTIC |
| `Best_Definition` | Which RUSTIC criterion matched best |
| `RUSTIC_Match` | "Yes" for all rows (only matches are saved) |
| `Score_Representative` … `Score_Comparability` | Per-criterion scores |

Additional per-criterion sheets show top matches sorted by each definition score.

---

## Notes

- The pipeline runs **indefinitely** against the full OpenAlex catalogue until the
  target is reached. For 400 matches from millions of documents, expect several hours
  depending on network speed and how many documents are open-access PDFs.
- A **checkpoint is saved every 25 matches**, so you can safely stop and restart.
  On restart, duplicate prevention is in-memory only; to avoid re-processing the
  same works, keep the pipeline running in a single session.
- Documents that time out, return no PDF, or yield empty text are silently skipped.

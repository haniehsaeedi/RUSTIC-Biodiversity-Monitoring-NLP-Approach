"""
train.py — Train the RUSTIC classifier and definition scorer.

Inputs  : Positive_RUSTIC_docs/  (local PDFs)
          Negative_RUSTIC.xlsx   (URLs, column 'link')
Outputs : models/rustic_pipeline.joblib
          models/rustic_scorer.joblib

Run once before pipeline.py:
    python train.py
"""

import os
import shutil
import tempfile
from collections import Counter

import joblib
import nltk
import pandas as pd
import requests
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from tqdm import tqdm

import config
from model_utils import (
    TfidfCosineScorer,
    cleanup_text,
    download_to_temp,
    extract_text_from_pdf_path,
)

# ── NLTK resources ────────────────────────────────────────────────────────────
try:
    _ = stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))
STEMMER   = PorterStemmer()


# ─────────────────────────────────────────────────────────────────────────────
# Cross-validated threshold tuning  (identical to notebook Cell 1)
# ─────────────────────────────────────────────────────────────────────────────

def crossval_fit_and_threshold(
    model,
    X: list[str],
    y: list[int],
    thresholds: list[float] | None = None,
    folds: int = 5,
) -> tuple:
    if thresholds is None:
        thresholds = [i / 100 for i in range(30, 71, 5)]
    skf     = StratifiedKFold(n_splits=min(folds, max(2, sum(y))),
                               shuffle=True, random_state=config.RANDOM_STATE)
    best_th = 0.5
    best_f1 = -1.0

    for th in thresholds:
        f1s = []
        for train_idx, val_idx in skf.split(X, y):
            X_tr = [X[i] for i in train_idx]
            y_tr = [y[i] for i in train_idx]
            X_vl = [X[i] for i in val_idx]
            y_vl = [y[i] for i in val_idx]
            model.fit(X_tr, y_tr)
            probs = model.predict_proba(X_vl)[:, 1]
            preds = (probs >= th).astype(int)
            f1s.append(f1_score(y_vl, preds, zero_division=0))
        mean_f1 = sum(f1s) / len(f1s)
        if mean_f1 > best_f1:
            best_f1, best_th = mean_f1, th

    model.fit(X, y)
    return model, best_th, best_f1


# ─────────────────────────────────────────────────────────────────────────────
# Main training routine
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  RUSTIC Classifier — Training")
    print("=" * 60)

    # ── 1. Positive documents (local PDFs) ────────────────────────────────────
    if not os.path.isdir(config.POSITIVE_PDF_DIR):
        raise FileNotFoundError(
            f"Positive PDF folder not found: '{config.POSITIVE_PDF_DIR}'"
        )
    positive_files = [
        os.path.join(config.POSITIVE_PDF_DIR, fn)
        for fn in os.listdir(config.POSITIVE_PDF_DIR)
        if fn.lower().endswith(".pdf")
    ]
    print(f"\nFound {len(positive_files)} positive PDF files.")

    pos_texts, pos_titles = [], []
    for p in tqdm(positive_files, desc="Extracting positives"):
        pos_texts.append(cleanup_text(extract_text_from_pdf_path(p)))
        pos_titles.append(os.path.basename(p))

    # ── 2. Negative documents (download from Excel) ───────────────────────────
    neg_texts, neg_titles = [], []
    if not os.path.exists(config.NEGATIVE_EXCEL):
        raise FileNotFoundError(
            f"Negative Excel not found: '{config.NEGATIVE_EXCEL}'"
        )

    neg_df = pd.read_excel(config.NEGATIVE_EXCEL)
    if config.NEG_LINK_COL not in neg_df.columns:
        raise ValueError(
            f"Negative Excel is missing column '{config.NEG_LINK_COL}'"
        )
    neg_links = neg_df[config.NEG_LINK_COL].dropna().astype(str).unique().tolist()
    print(f"\nAttempting to download {len(neg_links)} negative links ...")

    tmpdir = tempfile.mkdtemp(prefix="negpdf_")
    try:
        for link in tqdm(neg_links, desc="Downloading negatives"):
            path = download_to_temp(link, tmpdir,
                                    timeout=config.HTTP_TIMEOUT,
                                    user_agent=config.HTTP_USER_AGENT)
            if path and os.path.exists(path) and path.lower().endswith(".pdf"):
                neg_texts.append(cleanup_text(extract_text_from_pdf_path(path)))
                neg_titles.append(os.path.basename(path))
            else:
                try:
                    r = requests.get(link, timeout=config.HTTP_TIMEOUT,
                                     headers={"User-Agent": config.HTTP_USER_AGENT})
                    if "text/html" in r.headers.get("Content-Type", ""):
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(r.text, "lxml")
                        neg_texts.append(cleanup_text(soup.get_text(separator=" ")))
                        neg_titles.append(link)
                except Exception:
                    continue
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    # ── 3. Build training set ─────────────────────────────────────────────────
    texts  = pos_texts + neg_texts
    labels = [1] * len(pos_texts) + [0] * len(neg_texts)
    print(f"\nLabel counts: {Counter(labels)}")
    if sum(labels) == 0 or len(labels) - sum(labels) == 0:
        raise RuntimeError("Need both positive and negative examples to train.")

    # ── 4. Train TF-IDF + Logistic Regression ────────────────────────────────
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_df=0.95,
                                   sublinear_tf=True)),
        ("clf",   LogisticRegression(max_iter=500, class_weight="balanced",
                                      solver="liblinear",
                                      random_state=config.RANDOM_STATE)),
    ])
    print("\nTraining classifier with cross-validated threshold tuning ...")
    pipeline, best_threshold, best_cv_f1 = crossval_fit_and_threshold(
        pipeline, texts, labels, folds=5
    )
    print(f"  Best CV F1: {best_cv_f1:.3f}  |  Chosen threshold: {best_threshold:.2f}")

    os.makedirs(config.MODEL_DIR, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "threshold": best_threshold}, config.MODEL_PATH)
    print(f"  Saved model  → {config.MODEL_PATH}")

    # ── 5. Build and save the scorer ─────────────────────────────────────────
    scorer = TfidfCosineScorer().fit(texts, list(config.RUSTIC_DEFINITIONS.values()))
    joblib.dump({"scorer": scorer, "definitions": config.RUSTIC_DEFINITIONS},
                config.SCORER_PATH)
    print(f"  Saved scorer → {config.SCORER_PATH}")

    print("\n✅  Training complete. You can now run pipeline.py.")


if __name__ == "__main__":
    main()

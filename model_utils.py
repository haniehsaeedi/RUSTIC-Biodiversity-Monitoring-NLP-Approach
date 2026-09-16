"""
model_utils.py — Shared classes and helpers used by both train.py and pipeline.py.

Keeping TfidfCosineScorer here ensures joblib/pickle can always locate the class
regardless of which script is the entry point (__main__).
"""

import math
import os
import re

import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf_path(path: str) -> str:
    try:
        from pdfminer.high_level import extract_text as pdf_extract_text
        return pdf_extract_text(path) or ""
    except Exception:
        return ""


def cleanup_text(txt: str) -> str:
    if not txt:
        return ""
    txt = re.sub(r"\s+", " ", txt)
    txt = re.split(r"\n?references\b|\n?bibliography\b", txt, flags=re.I)[0]
    return txt.strip()


def simple_tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return [t for t in text.split() if t]


def text_to_windows(text: str, window_words: int = 250, step_words: int = 125) -> list[str]:
    words   = text.split()
    windows = []
    i = 0
    while i < len(words):
        window = words[i : i + window_words]
        if not window:
            break
        windows.append(" ".join(window))
        if i + window_words >= len(words):
            break
        i += step_words
    return windows


def bm25_best_window_score(def_text: str, doc_text: str) -> tuple[float, str]:
    from rank_bm25 import BM25Okapi
    windows = text_to_windows(doc_text)
    if not windows:
        return 0.0, ""
    tokenized = [simple_tokenize(w) for w in windows]
    # BM25 raises ZeroDivisionError if every window tokenizes to an empty list
    # (happens with PDFs that extract as only whitespace, numbers, or non-Latin text)
    tokenized = [t if t else ["_"] for t in tokenized]
    bm25      = BM25Okapi(tokenized)
    scores    = bm25.get_scores(simple_tokenize(def_text))
    best_idx  = int(max(range(len(scores)), key=lambda i: scores[i])) if len(scores) else 0
    return (float(scores[best_idx]) if len(scores) else 0.0,
            windows[best_idx] if windows else "")


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF cosine scorer  (must live here so joblib pickle always finds it)
# ─────────────────────────────────────────────────────────────────────────────

class TfidfCosineScorer:
    """Word + character n-gram TF-IDF cosine scorer for RUSTIC definition matching."""

    def __init__(self):
        self.word_vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True)
        self.char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 5),
                                         min_df=1, sublinear_tf=True)
        self.fitted = False

    def fit(self, docs: list[str], definitions: list[str]) -> "TfidfCosineScorer":
        corpus = docs + definitions
        self.word_vec.fit(corpus)
        self.char_vec.fit(corpus)
        self.fitted = True
        return self

    def best_window(self, definition: str, doc_text: str) -> tuple[float, str]:
        windows = text_to_windows(doc_text)
        if not windows:
            return 0.0, ""
        def_word = normalize(self.word_vec.transform([definition]))
        def_char = normalize(self.char_vec.transform([definition]))
        win_word = normalize(self.word_vec.transform(windows))
        win_char = normalize(self.char_vec.transform(windows))
        sims = (
            (win_word @ def_word.T).toarray().ravel() * 0.6
            + (win_char @ def_char.T).toarray().ravel() * 0.4
        )
        best_idx = int(max(range(len(sims)), key=lambda i: sims[i]))
        return float(sims[best_idx]), windows[best_idx]


# ─────────────────────────────────────────────────────────────────────────────
# Download helper
# ─────────────────────────────────────────────────────────────────────────────

def download_to_temp(url: str, tmpdir: str, timeout: int = 20,
                     user_agent: str = "RUSTIC Pipeline") -> str | None:
    """
    Download a URL to a temp file.
    - PDF response  → saved as .pdf
    - HTML response → scan for embedded PDF link; if found download it;
                      otherwise save page text as .txt
    - Other         → saved raw
    Returns local file path, or None on any failure.
    """
    try:
        headers = {"User-Agent": user_agent}
        r = requests.get(url, timeout=timeout, stream=True,
                         allow_redirects=True, headers=headers)
        r.raise_for_status()
        content_type = r.headers.get("Content-Type", "").lower()

        if "pdf" in content_type or url.lower().endswith(".pdf"):
            fname  = os.path.basename(url.split("?")[0]) or "download.pdf"
            target = os.path.join(tmpdir, fname)
            with open(target, "wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
            return target

        if "html" in content_type:
            from bs4 import BeautifulSoup
            from urllib.parse import urljoin
            soup     = BeautifulSoup(r.text, "lxml")
            pdf_href = None
            for a in soup.find_all("a", href=True):
                if ".pdf" in a["href"].lower():
                    pdf_href = a["href"]
                    break
            if pdf_href:
                if pdf_href.startswith("//"):
                    pdf_href = "https:" + pdf_href
                elif pdf_href.startswith("/"):
                    pdf_href = urljoin(url, pdf_href)
                return download_to_temp(pdf_href, tmpdir, timeout, user_agent)
            txt   = soup.get_text(separator=" ")
            fname = (os.path.basename(url.split("?")[0]) or "page") + ".txt"
            dest  = os.path.join(tmpdir, fname)
            with open(dest, "w", encoding="utf-8") as f:
                f.write(txt)
            return dest

        fname  = os.path.basename(url.split("?")[0]) or "download.bin"
        target = os.path.join(tmpdir, fname)
        with open(target, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    f.write(chunk)
        return target

    except Exception:
        return None
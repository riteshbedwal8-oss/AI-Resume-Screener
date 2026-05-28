# src/vectorizer.py
# Purpose: Convert text into TF-IDF vectors
# Replaces sentence-transformers to avoid torch dependency

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)


def fit_and_encode(all_texts: list) -> np.ndarray:
    """
    Fit vectorizer on ALL texts at once and return all vectors.
    Call this with: resume_texts + [jd_text]  (JD always last)
    Returns numpy array of shape (N, features)
    """
    global _vectorizer
    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    matrix = _vectorizer.fit_transform(all_texts).toarray()
    return matrix


def get_embedding(text: str) -> np.ndarray:
    """Transform single text using already fitted vectorizer."""
    global _vectorizer
    try:
        return _vectorizer.transform([text]).toarray()[0]
    except Exception:
        _vectorizer.fit([text])
        return _vectorizer.transform([text]).toarray()[0]


def get_batch_embeddings(texts: list) -> np.ndarray:
    """Transform multiple texts using already fitted vectorizer."""
    global _vectorizer
    try:
        return _vectorizer.transform(texts).toarray()
    except Exception:
        return _vectorizer.fit_transform(texts).toarray()
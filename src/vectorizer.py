# src/vectorizer.py
# Purpose: Convert text into TF-IDF vectors

from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

_is_fitted = False


def get_batch_embeddings(texts: list) -> np.ndarray:
    """Fit vectorizer on all texts and return their vectors."""
    global _vectorizer, _is_fitted
    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    _is_fitted = True
    return _vectorizer.fit_transform(texts).toarray()


def get_embedding(text: str) -> np.ndarray:
    """Transform a single text using already-fitted vectorizer."""
    global _vectorizer, _is_fitted
    if not _is_fitted:
        _vectorizer.fit([text])
        _is_fitted = True
    return _vectorizer.transform([text]).toarray()[0]
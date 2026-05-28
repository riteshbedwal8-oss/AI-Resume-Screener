# src/vectorizer.py
# Purpose: Convert preprocessed text into numerical vectors using TF-IDF
# (Replaces sentence-transformers to avoid torch dependency)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Global vectorizer instance
_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2)
)

_is_fitted = False


def get_embedding(text: str) -> np.ndarray:
    """Get TF-IDF vector for a single text."""
    global _is_fitted
    if not _is_fitted:
        _vectorizer.fit([text])
        _is_fitted = True
    return _vectorizer.transform([text]).toarray()[0]


def get_batch_embeddings(texts: list) -> np.ndarray:
    """Get TF-IDF vectors for a list of texts."""
    global _vectorizer, _is_fitted
    _vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words='english',
        ngram_range=(1, 2)
    )
    embeddings = _vectorizer.fit_transform(texts).toarray()
    _is_fitted = True
    return embeddings
# src/vectorizer.py
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
    all_texts = resume_texts + [jd_text]  (JD always last)
    """
    matrix = _vectorizer.fit_transform(all_texts).toarray()
    return matrix


def get_embedding(text: str) -> np.ndarray:
    """Transform single text using already fitted vectorizer."""
    return _vectorizer.transform([text]).toarray()[0]


def get_batch_embeddings(texts: list) -> np.ndarray:
    """Transform multiple texts using already fitted vectorizer."""
    return _vectorizer.transform(texts).toarray()
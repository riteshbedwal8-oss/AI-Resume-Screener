# src/vectorizer.py
# Uses sentence-transformers with lazy loading to save memory

from sentence_transformers import SentenceTransformer
import numpy as np

_model = None

def _get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def fit_and_encode(all_texts: list) -> np.ndarray:
    """
    Encode all texts together using sentence-transformers.
    all_texts = resume_texts + [jd_text]  (JD always last)
    Returns numpy array of shape (N, 384)
    """
    model = _get_model()
    embeddings = model.encode(all_texts, convert_to_numpy=True, show_progress_bar=False)
    return embeddings


def get_embedding(text: str) -> np.ndarray:
    """Encode a single text."""
    model = _get_model()
    return model.encode(text, convert_to_numpy=True)


def get_batch_embeddings(texts: list) -> np.ndarray:
    """Encode multiple texts."""
    model = _get_model()
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
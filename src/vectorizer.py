# src/vectorizer.py
from sentence_transformers import SentenceTransformer
import numpy as np

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)


def fit_and_encode(all_texts: list) -> np.ndarray:
    """
    Encode all texts together.
    all_texts = resume_texts + [jd_text]
    Returns numpy array of shape (N, 384)
    """
    embeddings = model.encode(all_texts, convert_to_numpy=True)
    return embeddings


def get_embedding(text: str) -> np.ndarray:
    return model.encode(text, convert_to_numpy=True)


def get_batch_embeddings(texts: list) -> np.ndarray:
    return model.encode(texts, convert_to_numpy=True)
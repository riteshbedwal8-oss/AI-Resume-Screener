# src/vectorizer.py
# Purpose: Convert preprocessed text into numerical vectors using Sentence Transformers

from sentence_transformers import SentenceTransformer
import numpy as np

# Load the model once. 'all-MiniLM-L6-v2' is fast and accurate.
# On first run, it downloads ~90MB automatically.
MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

print(f"[INFO] Embedding model '{MODEL_NAME}' loaded.")


def get_embedding(text: str) -> np.ndarray:
    """
    Convert a single text string into a dense vector.
    Output: numpy array of shape (384,) for MiniLM
    """
    embedding = model.encode(text, convert_to_numpy=True)
    return embedding


def get_batch_embeddings(texts: list) -> np.ndarray:
    """
    Convert multiple texts at once (faster than one-by-one).
    Output: numpy array of shape (N, 384) where N = number of texts
    """
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    return embeddings


# --- Quick test ---
if __name__ == "__main__":
    samples = [
        "Python developer with machine learning experience",
        "Data scientist skilled in deep learning and NLP"
    ]
    vecs = get_batch_embeddings(samples)
    print(f"Embedding shape: {vecs.shape}")  # Should be (2, 384)
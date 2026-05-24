# src/matcher.py
# Purpose: Compute cosine similarity between resume and job description embeddings

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def compute_similarity(resume_embedding: np.ndarray,
                        jd_embedding: np.ndarray) -> float:
    """
    Cosine similarity between two vectors.
    Score ranges from 0.0 (no match) to 1.0 (perfect match).
    We multiply by 100 to show as a percentage.
    """
    # Reshape to 2D as sklearn expects (1, N) arrays
    r = resume_embedding.reshape(1, -1)
    j = jd_embedding.reshape(1, -1)

    score = cosine_similarity(r, j)[0][0]
    return round(float(score) * 100, 2)   # e.g., 78.34


def score_all_resumes(resume_embeddings: np.ndarray,
                      jd_embedding: np.ndarray) -> list:
    """
    Score every resume against the job description.
    Returns a list of float scores, one per resume.
    """
    scores = []
    for emb in resume_embeddings:
        score = compute_similarity(emb, jd_embedding)
        scores.append(score)
    return scores
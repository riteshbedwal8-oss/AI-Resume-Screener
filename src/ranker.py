# src/ranker.py
# Purpose: Sort candidates by score and apply shortlisting threshold

import pandas as pd


def rank_candidates(candidate_names: list,
                    scores: list,
                    skill_data: list,
                    threshold: float = 50.0) -> pd.DataFrame:
    """
    Build a ranked DataFrame of candidates.

    Args:
        candidate_names: list of resume file names
        scores: list of similarity scores (0–100)
        skill_data: list of dicts with matched/missing skills per resume
        threshold: minimum score to be considered 'Shortlisted'

    Returns:
        A pandas DataFrame sorted by score (descending)
    """
    rows = []
    for i, name in enumerate(candidate_names):
        rows.append({
            "Candidate": name,
            "Match Score (%)": scores[i],
            "Matched Skills": ", ".join(skill_data[i]["matched"]) or "None",
            "Missing Skills": ", ".join(skill_data[i]["missing"]) or "None",
            "Status": "✅ Shortlisted" if scores[i] >= threshold else "❌ Rejected",
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("Match Score (%)", ascending=False)   # Highest score first
    df["Rank"] = range(1, len(df) + 1)
    df = df[["Rank", "Candidate", "Match Score (%)", "Status",
             "Matched Skills", "Missing Skills"]]
    return df.reset_index(drop=True)
# src/skill_extractor.py
# Purpose: Extract and compare skills between resume and job description

import re

# A curated list of common tech skills — extend this freely!
SKILL_DATABASE = [
    # Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "r", "scala",
    "go", "rust", "kotlin", "swift", "php", "ruby",
    # ML / AI
    "machine learning", "deep learning", "natural language processing",
    "computer vision", "reinforcement learning", "neural network",
    "tensorflow", "pytorch", "keras", "scikit-learn", "hugging face",
    "sentence transformers", "bert", "gpt", "llm",
    # Data
    "sql", "nosql", "mongodb", "postgresql", "mysql", "sqlite",
    "pandas", "numpy", "matplotlib", "seaborn", "tableau", "power bi",
    # Cloud / DevOps
    "amazon web services", "google cloud platform", "azure", "docker",
    "kubernetes", "ci/cd", "jenkins", "terraform", "git", "github",
    # Web
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "html", "css", "rest api", "graphql",
    # General
    "agile", "scrum", "communication", "leadership", "project management",
    "data analysis", "statistics", "excel", "problem solving",
]


def extract_skills(text: str) -> list:
    """
    Find all skills from SKILL_DATABASE that appear in the given text.
    Uses whole-word matching to avoid false positives (e.g. 'r' matching everywhere).
    """
    text_lower = text.lower()
    found = []
    for skill in SKILL_DATABASE:
        # \b = word boundary — ensures we match whole words only
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            found.append(skill)
    return found


def get_skill_gap(resume_skills: list, jd_skills: list) -> dict:
    """
    Compare resume skills vs job description skills.
    Returns:
      - matched: skills present in both
      - missing: skills in JD but not in resume
    """
    resume_set = set(resume_skills)
    jd_set = set(jd_skills)

    matched = list(resume_set & jd_set)      # Intersection
    missing = list(jd_set - resume_set)      # In JD but not resume

    return {
        "matched": sorted(matched),
        "missing": sorted(missing),
        "match_count": len(matched),
        "total_jd_skills": len(jd_set),
    }


# --- Quick test ---
if __name__ == "__main__":
    resume_text = "I have experience in Python, machine learning, and AWS."
    jd_text = "We need Python, machine learning, deep learning, and Docker."

    r_skills = extract_skills(resume_text)
    j_skills = extract_skills(jd_text)
    gap = get_skill_gap(r_skills, j_skills)

    print("Resume skills:", r_skills)
    print("JD skills:", j_skills)
    print("Gap analysis:", gap)
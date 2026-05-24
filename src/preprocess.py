# src/preprocess.py
# Purpose: Clean and normalize raw text for NLP processing

import re
import spacy
import nltk
from nltk.corpus import stopwords

# Load spaCy model once at module level (expensive to reload)
nlp = spacy.load("en_core_web_sm")

# Load NLTK stopwords
STOP_WORDS = set(stopwords.words("english"))


def clean_text(text: str) -> str:
    """
    Step 1: Basic cleaning
    - Lowercase everything
    - Remove special characters and extra whitespace
    - Keep only letters, numbers, and spaces
    """
    text = text.lower()                          # "Python Developer" → "python developer"
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text) # Remove @, #, !, etc.
    text = re.sub(r"\s+", " ", text)             # Collapse multiple spaces
    return text.strip()


def remove_stopwords(text: str) -> str:
    """
    Step 2: Remove stopwords
    Words like 'the', 'is', 'and' carry no useful information for matching.
    """
    words = text.split()
    filtered = [w for w in words if w not in STOP_WORDS]
    return " ".join(filtered)


def lemmatize_text(text: str) -> str:
    """
    Step 3: Lemmatization via spaCy
    Reduces words to their root form:
    'working' → 'work', 'managed' → 'manage', 'technologies' → 'technology'
    This ensures 'develop' and 'developing' are treated the same.
    """
    doc = nlp(text)
    lemmas = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
    return " ".join(lemmas)


def normalize_skills(text: str) -> str:
    """
    Step 4: Skill normalization
    Standardize common variations of the same skill:
    'ml' → 'machine learning', 'js' → 'javascript', etc.
    Extend this dictionary with more mappings as needed.
    """
    skill_map = {
        r"\bml\b": "machine learning",
        r"\bai\b": "artificial intelligence",
        r"\bdl\b": "deep learning",
        r"\bnlp\b": "natural language processing",
        r"\bjs\b": "javascript",
        r"\bts\b": "typescript",
        r"\bpy\b": "python",
        r"\bk8s\b": "kubernetes",
        r"\boci\b": "oracle cloud",
        r"\bgcp\b": "google cloud platform",
        r"\baws\b": "amazon web services",
        r"\bsql\b": "structured query language",
        r"\bapi\b": "application programming interface",
        r"\bci/cd\b": "continuous integration continuous deployment",
    }
    for pattern, replacement in skill_map.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text


def full_preprocess(text: str) -> str:
    """
    Master pipeline: runs all steps in order.
    This is what you call from other modules.
    """
    text = normalize_skills(text)   # Normalize FIRST (before lowercasing strips context)
    text = clean_text(text)
    text = remove_stopwords(text)
    text = lemmatize_text(text)
    return text


# --- Quick test ---
if __name__ == "__main__":
    sample = """
    Experienced ML Engineer with 5+ years in AI/DL and NLP.
    Proficient in Python, JS, and AWS. Managed CI/CD pipelines.
    """
    print("Original:\n", sample)
    print("\nProcessed:\n", full_preprocess(sample))
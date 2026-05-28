<div align="center">

# 🎯 ResumeIQ — AI Resume Screener

### An intelligent resume screening system powered by NLP & Machine Learning

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.45.0-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![spaCy](https://img.shields.io/badge/spaCy-NLP-09A3D5?style=for-the-badge)](https://spacy.io)

### 🌐 [Live Demo](https://ai-resume-screener-rchurmao2c8m4hi2jpzvt8.streamlit.app/) &nbsp;·&nbsp; 
</div>

---

## 📌 About The Project

**ResumeIQ** is an end-to-end AI-powered Resume Screening System that helps recruiters save hours of manual work. Upload multiple resumes, paste a Job Description, and get instant rankings, ATS scores, skill gap analysis, and actionable improvement suggestions — all in one beautiful dashboard.

---

## 🚀 Features

| Feature | Description |
|---|---|
| 📂 **Multi-Resume Upload** | Upload multiple PDF or DOCX resumes at once |
| 📝 **JD Matching** | Paste any Job Description and match against all resumes |
| 🏆 **Smart Ranking** | Candidates ranked by combined ATS + quality score |
| 🔍 **19-Point ATS Audit** | Deep audit across content, format, skills, sections & style |
| 🛠 **Skill Gap Analysis** | Shows exactly which skills match and which are missing |
| 📊 **Visual Dashboards** | Beautiful interactive charts powered by Plotly |
| ⬇️ **CSV Export** | Download the full screening report instantly |

---

## 🧠 Two-Tier Scoring Model

```
Final Score  =  Tier 1 (30%)  +  Tier 2 (70%)
```

- **Tier 1** — Semantic similarity between resume & job description using Sentence Transformers
- **Tier 2** — 19-point audit engine checking Content, Sections, Skills, Format & Style

### 📊 Grade Scale

| Score | Grade | Result |
|---|---|---|
| 85–100 | 🟢 A | Excellent — Shortlisted |
| 70–84 | 🟩 B | Good — Shortlisted |
| 55–69 | 🟡 C | Average — Needs Review |
| 40–54 | 🟠 D | Below Average — Rejected |
| 0–39 | 🔴 F | Poor — Rejected |

---

## 🛠 Tech Stack

```
Language      →  Python 3.11
Framework     →  Streamlit
NLP           →  spaCy, NLTK, Sentence Transformers
ML            →  Scikit-learn, Cosine Similarity
PDF Parsing   →  pdfplumber, pdfminer.six
DOCX Parsing  →  python-docx
Charts        →  Plotly
Deployment    →  Streamlit Cloud
```

---

## ⚙️ Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/riteshbedwal8-oss/AI-Resume-Screener.git
cd AI-Resume-Screener

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

---

## 📁 Project Structure

```
AI-Resume-Screener/
├── app.py                  ← Main Streamlit app
├── requirements.txt        ← Dependencies
├── runtime.txt             ← Python version
└── src/
    ├── parser.py           ← PDF & DOCX parser
    ├── preprocess.py       ← Text preprocessing
    ├── vectorizer.py       ← Text vectorization
    ├── matcher.py          ← Cosine similarity scoring
    ├── skill_extractor.py  ← NLP skill extraction
    └── ranker.py           ← Candidate ranking
```

---

## 👨‍💻 Author

**Ritesh Bedwal**

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/your-linkedin-here)

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

🌐 **[Try the Live App →](https://ai-resume-screener-rchurmao2c8m4hi2jpzvt8.streamlit.app/)**

</div>

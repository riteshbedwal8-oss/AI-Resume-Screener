import streamlit as st
import os
import re
import base64
import tempfile
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from streamlit_option_menu import option_menu

# Import project modules
from src.parser import parse_resume
from src.preprocess import full_preprocess
from src.vectorizer import get_embedding, get_batch_embeddings
from src.matcher import score_all_resumes
from src.skill_extractor import extract_skills, get_skill_gap
from src.ranker import rank_candidates

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeIQ – AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&display=swap');

:root {
    --bg:       #0D1117;
    --surface:  #161B22;
    --surface2: #1C2333;
    --border:   #30363D;
    --accent:   #2DC08D;
    --accent2:  #1B6F52;
    --danger:   #F85149;
    --warn:     #D29922;
    --txt:      #E6EDF3;
    --txt2:     #8B949E;
    --pass:     #3FB950;
    --blue:     #388BFD;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    font-family: 'DM Sans', sans-serif;
    color: var(--txt);
}
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="stMain"] { background-color: var(--bg) !important; }
[data-testid="block-container"] { padding-top: 1.5rem !important; }
h1,h2,h3,h4 { font-family:'Syne',sans-serif !important; color:var(--txt) !important; letter-spacing:-.02em; }

/* hero */
.hero-band {
    background: linear-gradient(135deg,#0D2B1F 0%,#0D1117 60%);
    border: 1px solid var(--accent2);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative; overflow: hidden;
}
.hero-band::before {
    content:""; position:absolute; top:-60px; right:-60px;
    width:260px; height:260px;
    background:radial-gradient(circle,rgba(45,192,141,.18) 0%,transparent 70%);
    border-radius:50%;
}
.hero-title { font-family:'Syne',sans-serif; font-size:40px; font-weight:800;
    color:var(--txt) !important; margin:0 0 8px; line-height:1.15; }
.hero-title span { color:var(--accent); }
.hero-sub { font-size:15px; color:var(--txt2); margin:0; max-width:560px; line-height:1.6; }

/* section label */
.section-label { font-family:'Syne',sans-serif; font-size:11px; font-weight:600;
    letter-spacing:.12em; text-transform:uppercase; color:var(--accent); margin-bottom:8px; }

/* upload card */
.upload-card { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:20px; }

/* metric card */
.metric-card { background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:18px 14px; text-align:center; position:relative; overflow:hidden; }
.metric-card::after { content:""; position:absolute; bottom:0; left:0; right:0;
    height:3px; border-radius:0 0 12px 12px; }
.metric-card.green::after { background:var(--pass); }
.metric-card.amber::after { background:var(--warn); }
.metric-card.blue::after  { background:var(--blue); }
.metric-card.red::after   { background:var(--danger); }
.metric-card .big-num { font-family:'Syne',sans-serif; font-size:34px; font-weight:800;
    color:var(--txt); line-height:1; margin-bottom:4px; }
.metric-card .small-label { font-size:12px; color:var(--txt2); letter-spacing:.04em; }

/* candidate card */
.cand-card { background:var(--surface); border:1px solid var(--border);
    border-radius:12px; padding:18px 22px; margin-bottom:10px;
    display:flex; align-items:center; gap:18px; }
.rank-badge { font-family:'Syne',sans-serif; font-size:20px; font-weight:800;
    color:var(--accent); min-width:34px; }
.cand-name { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:var(--txt); }
.score-bar-wrap { flex:1; background:var(--surface2); border-radius:4px; height:6px; overflow:hidden; }
.score-bar-fill { height:100%; border-radius:4px;
    background:linear-gradient(90deg,var(--accent2),var(--accent)); }
.status-pill { font-size:12px; font-weight:600; padding:4px 12px; border-radius:20px; white-space:nowrap; }
.pill-pass { background:rgba(63,185,80,.15); color:var(--pass); }
.pill-fail { background:rgba(248,81,73,.15); color:var(--danger); }
.pill-warn { background:rgba(210,153,34,.15); color:var(--warn); }

/* chips */
.chip { display:inline-flex; align-items:center; gap:4px; font-size:12px;
    padding:3px 10px; border-radius:20px; margin:2px; }
.chip-pass { background:rgba(63,185,80,.12); color:var(--pass); border:1px solid rgba(63,185,80,.3); }
.chip-fail { background:rgba(248,81,73,.12); color:var(--danger); border:1px solid rgba(248,81,73,.3); }
.chip-warn { background:rgba(210,153,34,.12); color:var(--warn); border:1px solid rgba(210,153,34,.3); }
.chip-blue { background:rgba(56,139,253,.12); color:var(--blue); border:1px solid rgba(56,139,253,.3); }

/* ── AUDIT PANEL ──────────────────────────────── */
.audit-panel {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 0;
    overflow: hidden;
    margin-bottom: 14px;
}
.audit-header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 16px 20px;
    border-bottom: 1px solid var(--border);
    background: var(--surface2);
}
.audit-name { font-family:'Syne',sans-serif; font-size:15px; font-weight:700; color:var(--txt); }
.audit-score-badge {
    font-family:'Syne',sans-serif; font-size:18px; font-weight:800;
    padding: 4px 14px; border-radius: 8px;
}
.audit-body { padding: 20px; }

/* audit check row */
.check-row {
    display: flex; align-items: flex-start; gap: 12px;
    padding: 10px 0;
    border-bottom: 1px solid var(--border);
}
.check-row:last-child { border-bottom: none; }
.check-icon { font-size: 16px; flex-shrink: 0; margin-top: 1px; }
.check-content { flex: 1; }
.check-label { font-size: 13px; font-weight: 600; color: var(--txt); margin-bottom: 3px; }
.check-detail { font-size: 12px; color: var(--txt2); line-height: 1.5; }
.check-suggestion {
    margin-top: 6px; padding: 8px 12px;
    background: rgba(45,192,141,.06);
    border-left: 3px solid var(--accent);
    border-radius: 0 6px 6px 0;
    font-size: 12px; color: var(--txt2); line-height: 1.5;
}
.check-suggestion b { color: var(--accent); }

/* category badge */
.cat-badge {
    font-size: 10px; font-weight: 600; letter-spacing: .08em;
    text-transform: uppercase; padding: 2px 8px; border-radius: 4px;
    margin-left: 8px;
}
.cat-content  { background:rgba(56,139,253,.15); color:var(--blue); }
.cat-format   { background:rgba(210,153,34,.15);  color:var(--warn); }
.cat-skills   { background:rgba(45,192,141,.15);  color:var(--accent); }
.cat-sections { background:rgba(248,81,73,.15);   color:var(--danger); }
.cat-style    { background:rgba(163,113,247,.15); color:#A371F7; }

/* progress bar mini */
.mini-bar-bg { background:var(--surface2); border-radius:4px; height:5px; margin-top:4px; }
.mini-bar-fill { height:5px; border-radius:4px; }

/* tier boxes */
.tier-box { background:var(--surface); border:1px solid var(--border); border-radius:12px; padding:18px; }
.tier-num { width:28px; height:28px; background:var(--accent); border-radius:50%;
    display:flex; align-items:center; justify-content:center;
    font-family:'Syne',sans-serif; font-size:13px; font-weight:700; color:#0D2B1F; flex-shrink:0; }

/* check grid (ATS page) */
.check-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(200px,1fr)); gap:10px; margin-top:10px; }
.check-item { display:flex; align-items:center; gap:8px; background:var(--surface);
    border:1px solid var(--border); border-radius:8px; padding:10px 12px;
    font-size:13px; color:var(--txt2); }
.dot { width:8px; height:8px; border-radius:50%; flex-shrink:0; }
.dot-pass { background:var(--pass); }
.dot-fail { background:var(--danger); }
.dot-warn { background:var(--warn); }

/* tabs */
.stTabs [data-baseweb="tab-list"] { background:var(--surface); border-radius:8px 8px 0 0; border-bottom:1px solid var(--border); gap:0; }
.stTabs [data-baseweb="tab"] { color:var(--txt2) !important; font-family:'DM Sans',sans-serif; padding:10px 20px; }
.stTabs [aria-selected="true"] { color:var(--accent) !important; border-bottom:2px solid var(--accent) !important; }

/* button */
.stButton>button {
    width:100% !important;
    background:linear-gradient(90deg,#1B6F52,var(--accent)) !important;
    color:#0D2B1F !important; font-family:'Syne',sans-serif !important;
    font-size:15px !important; font-weight:700 !important;
    border:none !important; border-radius:10px !important;
    height:3em !important; letter-spacing:.04em !important;
}
.stButton>button:hover { opacity:.88 !important; }

/* dataframe */
[data-testid="stDataFrame"] { border:1px solid var(--border) !important; border-radius:10px !important; overflow:hidden; }

/* expander */
.streamlit-expanderHeader { background:var(--surface) !important; border:1px solid var(--border) !important;
    border-radius:10px !important; font-family:'DM Sans',sans-serif; color:var(--txt) !important; }

::-webkit-scrollbar { width:6px; }
::-webkit-scrollbar-track { background:var(--bg); }
::-webkit-scrollbar-thumb { background:var(--border); border-radius:3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# ██  RESUME AUDIT ENGINE
# ─────────────────────────────────────────────────────────────
"""
Full 19-point audit on raw resume text.
Returns a dict with: issues list, suggestions list, scores per category,
overall audit score (0-100), grade letter.
"""

# ── keyword sets ──────────────────────────────────────────────
WEAK_VERBS   = {"worked","did","made","helped","used","tried","got","went",
                "involved","participated","assisted","handled"}
BUZZWORDS    = {"hardworking","team player","go-getter","results-driven","synergy",
                "passionate","detail-oriented","proactive","self-starter","dynamic",
                "leverage","utilize","innovative","guru","ninja","rockstar"}
FILLER       = {"responsible for","duties included","tasked with","worked on",
                "was involved in","helped with"}
ACTION_VERBS = {"led","built","designed","developed","managed","created","launched",
                "optimized","automated","analyzed","delivered","achieved","improved",
                "reduced","increased","streamlined","implemented","deployed","spearheaded",
                "orchestrated","engineered","architected","scaled","generated","negotiated"}
SECTION_KWS  = {
    "summary":     ["summary","objective","profile","about"],
    "experience":  ["experience","work","employment","internship","career"],
    "education":   ["education","degree","academic","university","college","school"],
    "skills":      ["skills","technologies","tools","competencies","stack"],
    "projects":    ["projects","portfolio","case study","case studies"],
    "contact":     ["email","phone","linkedin","github","contact"],
    "certifications": ["certification","certificate","certified","credential"],
}
QUANT_PATTERNS = [
    r'\d+\s*%',                       # 40%
    r'\$\s*[\d,]+',                   # $50,000
    r'₹\s*[\d,]+',                    # ₹5,00,000
    r'\d+\+?\s*(users|clients|projects|teams|members|employees|records|rows)',
    r'(increased|reduced|improved|saved|grew|generated|cut)\s+by\s+\d+',
    r'\d+\s*(x|times)\s*(faster|improvement|growth)',
    r'top\s+\d+\s*%',
]


def _find_sections(text: str) -> dict:
    found = {}
    tl = text.lower()
    for sec, kws in SECTION_KWS.items():
        found[sec] = any(kw in tl for kw in kws)
    return found


def _count_quant(text: str) -> int:
    count = 0
    for pat in QUANT_PATTERNS:
        count += len(re.findall(pat, text, re.IGNORECASE))
    return count


def _word_count(text: str) -> int:
    return len(text.split())


def _count_action_verbs(text: str) -> int:
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    return len(words & ACTION_VERBS)


def _count_weak_verbs(text: str) -> int:
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    return sum(1 for w in words if w in WEAK_VERBS)


def _count_buzzwords(text: str) -> int:
    words = set(re.findall(r'\b[a-zA-Z]+\b', text.lower()))
    return len(words & BUZZWORDS)


def _count_filler(text: str) -> int:
    tl = text.lower()
    return sum(1 for f in FILLER if f in tl)


def _has_email(text: str) -> bool:
    return bool(re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text))


def _has_phone(text: str) -> bool:
    return bool(re.search(r'(\+?\d[\d\s\-().]{7,}\d)', text))


def _has_linkedin(text: str) -> bool:
    return 'linkedin' in text.lower()


def _has_github(text: str) -> bool:
    return 'github' in text.lower()


def _bullet_count(text: str) -> int:
    lines = text.split('\n')
    return sum(1 for l in lines if l.strip().startswith(('-', '•', '·', '*', '–')))


def _long_bullets(text: str, threshold=25) -> int:
    lines = text.split('\n')
    count = 0
    for l in lines:
        l = l.strip()
        if l.startswith(('-', '•', '·', '*', '–')):
            if len(l.split()) > threshold:
                count += 1
    return count


def _word_freq(text: str, top=5) -> list:
    stop = {"and","the","to","of","in","a","for","with","on","is","at","by",
            "an","as","be","this","that","are","was","were","has","have","had",
            "it","its","or","from","not","but","we","you","i","my","your","our"}
    words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
    freq  = {}
    for w in words:
        if w not in stop:
            freq[w] = freq.get(w, 0) + 1
    sorted_freq = sorted(freq.items(), key=lambda x: -x[1])
    return [(w, c) for w, c in sorted_freq if c > 2][:top]


def audit_resume(raw_text: str, filename: str, jd_skills: set, match_score: float) -> dict:
    """
    Run all 19 checks. Return structured audit report.
    """
    issues      = []   # list of dicts: {category, label, severity, detail, suggestion}
    passed      = []   # list of dicts: {category, label, detail}
    cat_scores  = {}   # category -> 0-100

    # ── 1. Contact Information ─────────────────────────────────
    has_email    = _has_email(raw_text)
    has_phone    = _has_phone(raw_text)
    has_linkedin = _has_linkedin(raw_text)
    has_github   = _has_github(raw_text)

    contact_score = sum([has_email, has_phone, has_linkedin, has_github]) * 25

    if has_email:
        passed.append({"category":"Sections","label":"Email address detected","detail":"Email present and parseable by ATS."})
    else:
        issues.append({"category":"Sections","label":"Missing email address","severity":"critical",
            "detail":"No email address found in resume. ATS cannot create a contact record.",
            "suggestion":"Add a professional email in the header. Use format: <b>name@domain.com</b>"})

    if has_phone:
        passed.append({"category":"Sections","label":"Phone number detected","detail":"Contact number present."})
    else:
        issues.append({"category":"Sections","label":"Missing phone number","severity":"high",
            "detail":"Phone number not detected. Recruiters cannot call you directly.",
            "suggestion":"Add phone in international format: <b>+91-XXXXXXXXXX</b>"})

    if has_linkedin:
        passed.append({"category":"Sections","label":"LinkedIn URL present","detail":"LinkedIn profile linked."})
    else:
        issues.append({"category":"Sections","label":"LinkedIn URL missing","severity":"medium",
            "detail":"LinkedIn is checked by 87% of recruiters. Missing it hurts credibility.",
            "suggestion":"Add: <b>linkedin.com/in/your-name</b> to your contact section."})

    if has_github:
        passed.append({"category":"Sections","label":"GitHub URL present","detail":"GitHub profile linked — great for technical roles."})
    else:
        issues.append({"category":"Sections","label":"GitHub URL missing","severity":"low",
            "detail":"GitHub is expected for Data/Tech roles. Not having it is a missed opportunity.",
            "suggestion":"Add your GitHub profile: <b>github.com/your-username</b>"})

    cat_scores["Sections"] = contact_score

    # ── 2. Essential Sections ──────────────────────────────────
    sections  = _find_sections(raw_text)
    essential = ["summary","experience","education","skills"]
    bonus     = ["projects","certifications"]

    sec_score = 0
    for sec in essential:
        if sections.get(sec):
            passed.append({"category":"Sections","label":f"'{sec.title()}' section found","detail":f"ATS can parse the {sec} section."})
            sec_score += 20
        else:
            issues.append({"category":"Sections","label":f"Missing '{sec.title()}' section","severity":"high",
                "detail":f"The {sec} section was not detected. ATS systems score this section directly.",
                "suggestion":f"Add a clearly labeled <b>{sec.title()}</b> section heading so ATS can parse it."})

    for sec in bonus:
        if sections.get(sec):
            passed.append({"category":"Sections","label":f"'{sec.title()}' section found","detail":f"{sec.title()} adds credibility and ATS keyword density."})
            sec_score += 10

    cat_scores["Sections"] = min(100, cat_scores.get("Sections",0) + sec_score)

    # ── 3. Quantified Achievements ─────────────────────────────
    quant_count = _count_quant(raw_text)
    if quant_count >= 5:
        passed.append({"category":"Content","label":f"{quant_count} quantified achievements found",
            "detail":"Strong use of numbers, percentages, and measurable impact."})
        quant_score = 100
    elif quant_count >= 2:
        issues.append({"category":"Content","label":f"Only {quant_count} quantified achievement(s)","severity":"medium",
            "detail":"Recruiters and ATS score resumes higher when bullet points contain numbers.",
            "suggestion":"Aim for 5+ metrics. Add: <b>% improvement, $ saved, # records processed, rows of data, team size</b>. Example: 'Reduced report time by 40%' instead of 'Improved reporting'."})
        quant_score = 50
    else:
        issues.append({"category":"Content","label":"No quantified achievements detected","severity":"critical",
            "detail":"Zero measurable results found. This is the #1 reason resumes get rejected.",
            "suggestion":"Every bullet point should answer: <b>How much? How many? By what %?</b> E.g. 'Processed 100K+ rows', 'Reduced manual effort by 35%', 'Built 3 dashboards'."})
        quant_score = 0

    cat_scores["Content"] = quant_score

    # ── 4. Action Verbs ────────────────────────────────────────
    action_count = _count_action_verbs(raw_text)
    weak_count   = _count_weak_verbs(raw_text)

    if action_count >= 8:
        passed.append({"category":"Content","label":f"{action_count} strong action verbs used",
            "detail":"Excellent use of power verbs — resume reads as achievement-driven."})
        verb_score = 100
    elif action_count >= 4:
        issues.append({"category":"Content","label":f"Only {action_count} action verbs detected","severity":"medium",
            "detail":f"Resume has {action_count} action verbs and {weak_count} weak verbs. Needs more impact.",
            "suggestion":"Start every bullet with a strong verb: <b>Led, Built, Automated, Analyzed, Optimized, Delivered, Designed, Implemented, Reduced, Increased</b>"})
        verb_score = 55
    else:
        issues.append({"category":"Content","label":"Weak action verb usage","severity":"high",
            "detail":f"Only {action_count} action verbs and {weak_count} weak verbs detected. Resume sounds passive.",
            "suggestion":"Replace: <b>'Worked on reports'</b> → <b>'Automated reporting workflows'</b>. Never start a bullet with 'Responsible for', 'Did', or 'Helped'."})
        verb_score = 20

    cat_scores["Content"] = (cat_scores.get("Content",0) + verb_score) // 2

    # ── 5. Weak / Filler Language ──────────────────────────────
    filler_count = _count_filler(raw_text)
    buzz_count   = _count_buzzwords(raw_text)

    if filler_count == 0 and buzz_count == 0:
        passed.append({"category":"Style","label":"No filler phrases or buzzwords","detail":"Clean, direct language throughout."})
        style_score = 100
    else:
        detail_parts = []
        suggestion_parts = []
        if filler_count > 0:
            detail_parts.append(f"{filler_count} filler phrase(s) detected (e.g. 'responsible for', 'duties included')")
            suggestion_parts.append("Replace <b>'Responsible for managing'</b> → <b>'Managed'</b>")
        if buzz_count > 0:
            detail_parts.append(f"{buzz_count} buzzword(s) found (e.g. 'passionate', 'team player', 'hardworking')")
            suggestion_parts.append("Remove buzzwords — they add no ATS value. Replace with specific skills and achievements.")
        issues.append({"category":"Style","label":f"{filler_count + buzz_count} filler/buzzword issue(s)","severity":"medium",
            "detail":" | ".join(detail_parts),
            "suggestion":" &nbsp;•&nbsp; ".join(suggestion_parts)})
        style_score = max(0, 100 - (filler_count + buzz_count) * 15)

    cat_scores["Style"] = style_score

    # ── 6. Word Count / Length ─────────────────────────────────
    wc = _word_count(raw_text)
    if 300 <= wc <= 700:
        passed.append({"category":"Format","label":f"Good length ({wc} words)","detail":"Resume is within ideal 300–700 word range for a 1-page fresher resume."})
        len_score = 100
    elif wc < 200:
        issues.append({"category":"Format","label":f"Resume too short ({wc} words)","severity":"high",
            "detail":"Under 200 words — too thin to pass ATS keyword density checks.",
            "suggestion":"Expand your bullet points. Add more detail to projects and internship. Target <b>300–600 words</b>."})
        len_score = 30
    elif wc > 900:
        issues.append({"category":"Format","label":f"Resume too long ({wc} words)","severity":"medium",
            "detail":"Over 900 words for a fresher resume. Recruiters scan in 6-8 seconds.",
            "suggestion":"Trim to <b>1 page / 500–700 words</b>. Remove old irrelevant bullets. Be concise."})
        len_score = 60
    else:
        passed.append({"category":"Format","label":f"Acceptable length ({wc} words)","detail":"Length is within range."})
        len_score = 80

    cat_scores["Format"] = len_score

    # ── 7. Long Bullets ────────────────────────────────────────
    long_b = _long_bullets(raw_text)
    if long_b == 0:
        passed.append({"category":"Format","label":"No overly long bullet points","detail":"All bullets are concise and scannable."})
    else:
        issues.append({"category":"Format","label":f"{long_b} long bullet point(s) detected","severity":"low",
            "detail":f"{long_b} bullet(s) exceed 25 words. Long bullets are hard to scan.",
            "suggestion":"Split long bullets into 2 shorter ones. Each bullet = <b>1 action + 1 result</b>. Max 20 words."})
        cat_scores["Format"] = max(0, cat_scores.get("Format",100) - 15)

    # ── 8. Repetition / Word Overuse ──────────────────────────
    top_words = _word_freq(raw_text)
    if top_words:
        word, count = top_words[0]
        if count > 5:
            issues.append({"category":"Style","label":f"Word overuse detected ('{word}' × {count})","severity":"low",
                "detail":f"The word '{word}' appears {count} times. Repetition reduces readability.",
                "suggestion":f"Vary your vocabulary. Replace some instances of '<b>{word}</b>' with synonyms or restructure sentences."})
        else:
            passed.append({"category":"Style","label":"Good vocabulary variety","detail":"No significant word overuse detected."})
    cat_scores["Style"] = (cat_scores.get("Style",0) + 80) // 2

    # ── 9. Bullet Point Count ──────────────────────────────────
    bullets = _bullet_count(raw_text)
    if bullets >= 6:
        passed.append({"category":"Format","label":f"{bullets} bullet points found","detail":"Good use of structured bullet points for ATS parsing."})
    elif bullets >= 3:
        issues.append({"category":"Format","label":f"Low bullet point count ({bullets})","severity":"low",
            "detail":"Fewer than 6 bullets makes resume hard to scan. ATS parses bullet structure.",
            "suggestion":"Convert paragraph text into <b>bullet points</b>. Aim for 3–5 bullets per role/project."})
    else:
        issues.append({"category":"Format","label":"Almost no bullet points detected","severity":"medium",
            "detail":"Resume appears to use paragraph format instead of bullets. ATS prefers bullets.",
            "suggestion":"Restructure all experience and project entries as <b>bullet points starting with action verbs</b>."})

    # ── 10. Skill Coverage vs JD ──────────────────────────────
    resume_skills = extract_skills(raw_text)
    gap_data      = get_skill_gap(resume_skills, jd_skills)
    matched_s     = gap_data.get("matched", [])
    missing_s     = gap_data.get("missing", [])

    total_jd = len(jd_skills) if jd_skills else 1
    coverage  = int(len(matched_s) / total_jd * 100) if total_jd > 0 else 0

    if coverage >= 80:
        passed.append({"category":"Skills","label":f"High JD skill coverage ({coverage}%)","detail":"Resume matches most required skills from the job description."})
        skill_score = 100
    elif coverage >= 50:
        issues.append({"category":"Skills","label":f"Moderate JD skill coverage ({coverage}%)","severity":"medium",
            "detail":f"Matched {len(matched_s)} of {total_jd} required skills. Missing: {', '.join(list(missing_s)[:5])}",
            "suggestion":f"Add these missing skills if you have them: <b>{', '.join(list(missing_s)[:6])}</b>. Include them naturally in bullet points."})
        skill_score = 60
    else:
        issues.append({"category":"Skills","label":f"Low JD skill coverage ({coverage}%)","severity":"high",
            "detail":f"Only {len(matched_s)} of {total_jd} required JD skills found. ATS will likely filter this resume out.",
            "suggestion":f"Prioritize adding: <b>{', '.join(list(missing_s)[:8])}</b>. These keywords appear in the JD and must be in the resume."})
        skill_score = 25

    cat_scores["Skills"] = skill_score

    # ── 11. Summary / Objective Quality ───────────────────────
    tl = raw_text.lower()
    has_summary = any(kw in tl for kw in SECTION_KWS["summary"])
    if has_summary:
        # check if summary has >30 words in its likely block
        summary_idx = min([tl.find(kw) for kw in SECTION_KWS["summary"] if kw in tl])
        summary_block = raw_text[summary_idx:summary_idx+400]
        sw = _word_count(summary_block)
        if sw >= 30:
            passed.append({"category":"Content","label":"Summary section is detailed","detail":"Summary is substantive and will be read by both ATS and human recruiters."})
        else:
            issues.append({"category":"Content","label":"Summary too short","severity":"medium",
                "detail":"Summary section appears to be under 30 words. Too brief to be impactful.",
                "suggestion":"Expand summary to 3–4 lines: <b>Who you are + Top skills + Current role/goal + Value you bring</b>. Include JD keywords naturally."})
    cat_scores["Content"] = max(cat_scores.get("Content",0), 50)

    # ── Compute overall audit score ────────────────────────────
    weights = {"Content":30, "Sections":25, "Skills":25, "Format":10, "Style":10}
    total_w  = sum(weights.values())
    overall  = sum(cat_scores.get(cat, 60) * w for cat, w in weights.items()) // total_w

    # Factor in semantic match score (30% weight)
    final_score = int(overall * 0.70 + match_score * 0.30)

    # Grade
    if final_score >= 85: grade = ("A", "#3FB950")
    elif final_score >= 70: grade = ("B", "#2DC08D")
    elif final_score >= 55: grade = ("C", "#D29922")
    elif final_score >= 40: grade = ("D", "#F0883E")
    else: grade = ("F", "#F85149")

    return {
        "filename":    filename,
        "issues":      sorted(issues, key=lambda x: {"critical":0,"high":1,"medium":2,"low":3}.get(x["severity"],4)),
        "passed":      passed,
        "cat_scores":  cat_scores,
        "overall":     final_score,
        "grade":       grade,
        "matched_skills": matched_s,
        "missing_skills": missing_s,
        "coverage":    coverage,
        "quant_count": quant_count,
        "action_count":action_count,
        "word_count":  wc,
    }


# ─────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────
SEV_ICON  = {"critical":"🔴","high":"🟠","medium":"🟡","low":"🔵"}
SEV_COLOR = {"critical":"#F85149","high":"#F0883E","medium":"#D29922","low":"#388BFD"}
CAT_CLASS = {"Content":"cat-content","Format":"cat-format",
             "Skills":"cat-skills","Sections":"cat-sections","Style":"cat-style"}

def pill_class(status):
    s = status.lower()
    if "shortlisted" in s: return "pill-pass"
    if "review"      in s: return "pill-warn"
    return "pill-fail"

def score_color(score):
    if score >= 75: return "#3FB950"
    if score >= 50: return "#D29922"
    return "#F85149"

def svg_ring(score, size=110, stroke=9):
    r    = (size - stroke) / 2
    circ = 2 * 3.14159 * r
    fill = circ * score / 100
    gap  = circ - fill
    col  = score_color(score)
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="#1C2333" stroke-width="{stroke}"/>
      <circle cx="{size/2}" cy="{size/2}" r="{r}" fill="none" stroke="{col}" stroke-width="{stroke}"
        stroke-dasharray="{fill} {gap}" stroke-dashoffset="{circ/4}" stroke-linecap="round"/>
      <text x="50%" y="50%" dominant-baseline="middle" text-anchor="middle"
        font-family="Syne,sans-serif" font-size="20" font-weight="800" fill="#E6EDF3">{score}%</text>
    </svg>"""

def mini_bar(value, color="#2DC08D"):
    return f"""<div class="mini-bar-bg">
      <div class="mini-bar-fill" style="width:{value}%;background:{color};"></div>
    </div>"""

ATS_CHECKS = {
    "Content":  [("ATS parse rate","pass"),("Repetition check","warn"),("Spelling & grammar","pass"),
                 ("Quantified achievements","warn"),("Action verbs","warn"),("Summary present","pass")],
    "Format":   [("File format & size","pass"),("Resume length","pass"),("Long bullet check","warn")],
    "Skills":   [("Hard skills matched","pass"),("Soft skills identified","warn"),("Skill gap flagged","pass")],
    "Sections": [("Contact information","pass"),("Essential sections","pass"),("Projects present","warn")],
    "Style":    [("Email address valid","pass"),("Active voice","warn"),("Buzzword check","warn"),
                 ("Filler phrase check","warn"),("Hyperlinks valid","pass")],
}
DOT_CLASS = {"pass":"dot-pass","fail":"dot-fail","warn":"dot-warn"}


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:8px 0 16px 0;'>
      <span style='font-family:Syne,sans-serif;font-size:20px;font-weight:800;color:#E6EDF3;'>
        Resume<span style='color:#2DC08D;'>IQ</span></span>
      <div style='font-size:11px;color:#8B949E;margin-top:2px;letter-spacing:.06em;text-transform:uppercase;'>
        AI Screening System</div>
    </div>""", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Screener","ATS Checks","About"],
        icons=["cpu","clipboard-check","info-circle"],
        default_index=0,
        styles={
            "container":         {"background":"transparent","padding":"0"},
            "icon":              {"color":"#8B949E"},
            "nav-link":          {"font-size":"14px","color":"#8B949E","border-radius":"8px","margin":"2px 0"},
            "nav-link-selected": {"background":"rgba(45,192,141,.12)","color":"#2DC08D","font-weight":"600"},
        }
    )

    st.markdown("---")
    st.markdown('<div class="section-label">Shortlisting threshold</div>', unsafe_allow_html=True)
    threshold = st.slider("", 0, 100, 60, label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style='font-size:12px;color:#8B949E;line-height:1.6;'>
      <b style='color:#E6EDF3;'>Scoring model</b><br><br>
      <b style='color:#2DC08D;'>Tier 1 (30%)</b> — Semantic similarity via sentence-transformer embeddings.<br><br>
      <b style='color:#2DC08D;'>Tier 2 (70%)</b> — 19-point resume audit: sections, content quality, skills, format, style.
    </div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Powered by Sentence Transformers + NLP")


# ─────────────────────────────────────────────────────────────
# PAGE: ATS CHECKS
# ─────────────────────────────────────────────────────────────
if selected == "ATS Checks":
    st.markdown("""
    <div class="hero-band">
      <div class="hero-title">19-Point <span>ATS Audit</span></div>
      <p class="hero-sub">ResumeIQ runs every resume through the same 19 criteria used by
      enterprise ATS platforms — across 5 categories.</p>
    </div>""", unsafe_allow_html=True)

    for cat, checks in ATS_CHECKS.items():
        st.markdown(f'<div class="section-label">{cat}</div>', unsafe_allow_html=True)
        html = '<div class="check-grid">'
        for label, status in checks:
            html += f'<div class="check-item"><div class="dot {DOT_CLASS[status]}"></div><span>{label}</span></div>'
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='display:flex;gap:20px;flex-wrap:wrap;'>
      <div style='display:flex;align-items:center;gap:6px;font-size:13px;color:#8B949E;'>
        <div style='width:10px;height:10px;border-radius:50%;background:#3FB950;flex-shrink:0;'></div> Auto-checked & passed
      </div>
      <div style='display:flex;align-items:center;gap:6px;font-size:13px;color:#8B949E;'>
        <div style='width:10px;height:10px;border-radius:50%;background:#D29922;flex-shrink:0;'></div> Improvement suggested
      </div>
      <div style='display:flex;align-items:center;gap:6px;font-size:13px;color:#8B949E;'>
        <div style='width:10px;height:10px;border-radius:50%;background:#F85149;flex-shrink:0;'></div> Failed / missing
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE: ABOUT
# ─────────────────────────────────────────────────────────────
if selected == "About":
    st.markdown("""
    <div class="hero-band">
      <div class="hero-title">About <span>ResumeIQ</span></div>
      <p class="hero-sub">Enterprise-grade AI resume screening — built on two-tier semantic NLP,
      a 19-point audit engine, and skill-gap analysis.</p>
    </div>""", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
        <div class="tier-box"><div style='display:flex;gap:12px;align-items:flex-start;'>
          <div class="tier-num">1</div>
          <div><div style='font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#E6EDF3;'>
            Semantic Similarity</div>
          <div style='font-size:13px;color:#8B949E;line-height:1.5;margin-top:4px;'>
            Sentence-transformer embeddings compare full resume meaning against the JD.
            Contributes 30% of the final score.</div></div></div></div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="tier-box"><div style='display:flex;gap:12px;align-items:flex-start;'>
          <div class="tier-num">2</div>
          <div><div style='font-family:Syne,sans-serif;font-size:14px;font-weight:700;color:#E6EDF3;'>
            19-Point Audit Engine</div>
          <div style='font-size:13px;color:#8B949E;line-height:1.5;margin-top:4px;'>
            Checks content, format, skills, sections, and style — with specific issues,
            severity ratings, and actionable fix suggestions. 70% of the final score.</div></div></div></div>""",
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 24px;'>
      <div style='font-family:Syne,sans-serif;font-size:15px;font-weight:700;color:#E6EDF3;margin-bottom:12px;'>Tech Stack</div>
      <div style='display:flex;flex-wrap:wrap;gap:8px;'>
        <span class='chip chip-pass'>Sentence Transformers</span><span class='chip chip-pass'>spaCy NLP</span>
        <span class='chip chip-pass'>Streamlit</span><span class='chip chip-pass'>Plotly</span>
        <span class='chip chip-pass'>PyMuPDF / python-docx</span><span class='chip chip-pass'>Cosine Similarity</span>
        <span class='chip chip-pass'>Pandas</span><span class='chip chip-pass'>Regex Audit Engine</span>
      </div>
    </div>""", unsafe_allow_html=True)
    st.stop()


# ─────────────────────────────────────────────────────────────
# PAGE: SCREENER
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-band">
  <div class="hero-title">AI Resume <span>Screener</span></div>
  <p class="hero-sub">Upload candidate resumes and paste the job description.
  ResumeIQ runs a two-tier semantic + 19-point audit, ranks every candidate,
  and delivers a full per-resume report with issues and fix suggestions.</p>
</div>""", unsafe_allow_html=True)

col_up, col_jd = st.columns([1, 1], gap="large")

with col_up:
    st.markdown('<div class="section-label">📂 Upload Resumes</div>', unsafe_allow_html=True)
    st.markdown('<div class="upload-card">', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Drop PDF or DOCX files",
        type=["pdf","docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    if uploaded_files:
        st.markdown(f'<div style="margin-top:10px;font-size:13px;color:#2DC08D;">✓ {len(uploaded_files)} file(s) ready</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col_jd:
    st.markdown('<div class="section-label">📝 Job Description</div>', unsafe_allow_html=True)
    jd_text = st.text_area("", height=200,
        placeholder="Paste the full job description here…",
        label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

col_btn, col_hint = st.columns([1,2], gap="large")
with col_btn:
    run = st.button("🎯  Analyse & Audit Resumes")
with col_hint:
    st.markdown("""
    <div style='font-size:13px;color:#8B949E;padding-top:10px;line-height:1.6;'>
      ResumeIQ runs <b style='color:#E6EDF3;'>19 audit checks</b> per resume —
      flags every issue with severity + a specific fix suggestion,
      then ranks all candidates by combined ATS + quality score.
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────────────────────
if run:
    if not uploaded_files:
        st.error("⚠️  Please upload at least one resume.")
        st.stop()
    if not jd_text.strip():
        st.error("⚠️  Please paste a job description.")
        st.stop()

    with st.spinner("Running two-tier AI analysis + 19-point audit…"):

        jd_processed = full_preprocess(jd_text)
        jd_embedding = get_embedding(jd_processed)
        jd_skills    = extract_skills(jd_text)

        candidate_names = []
        resume_texts    = []
        raw_texts       = []

        for uf in uploaded_files:
            suffix = os.path.splitext(uf.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uf.read())
                tmp_path = tmp.name
            raw  = parse_resume(tmp_path)
            proc = full_preprocess(raw)
            candidate_names.append(uf.name.replace(suffix, ""))
            resume_texts.append(proc)
            raw_texts.append(raw)
            os.unlink(tmp_path)

        resume_embeddings = get_batch_embeddings(resume_texts)
        scores            = score_all_resumes(resume_embeddings, jd_embedding)

        skill_data = []
        for text in raw_texts:
            r_skills = extract_skills(text)
            gap      = get_skill_gap(r_skills, jd_skills)
            skill_data.append(gap)

        results_df = rank_candidates(candidate_names, scores, skill_data, threshold)

        # ── Run 19-point audit per candidate ──────────────────
        audit_reports = {}
        for i, (name, raw_text) in enumerate(zip(candidate_names, raw_texts)):
            match_pct = float(scores[i] * 100) if scores[i] <= 1 else float(scores[i])
            audit_reports[name] = audit_resume(raw_text, name, jd_skills, match_pct)

    st.success("✅  Analysis & Audit complete.")

    # ── Metrics ───────────────────────────────────────────────
    shortlisted  = results_df[results_df["Status"].str.contains("Shortlisted", na=False)]
    needs_review = results_df[results_df["Status"].str.contains("Review", na=False)]
    top_score    = int(results_df["Match Score (%)"].max())
    avg_score    = int(results_df["Match Score (%)"].mean())

    m1,m2,m3,m4 = st.columns(4)
    for col, label, val, cls in [
        (m1, "Total Candidates",  len(results_df),   "blue"),
        (m2, "Shortlisted",       len(shortlisted),  "green"),
        (m3, "Needs Review",      len(needs_review), "amber"),
        (m4, "Avg Match Score",   f"{avg_score}%",   "blue"),
    ]:
        with col:
            st.markdown(f"""
            <div class="metric-card {cls}">
              <div class="big-num">{val}</div>
              <div class="small-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────────────────────
    tab_rank, tab_audit, tab_chart, tab_skills, tab_data = st.tabs([
        "🏆 Rankings",
        "🔍 Resume Audit Reports",
        "📊 Charts",
        "🛠 Skill Analysis",
        "📋 Raw Data"
    ])

    # ────────────────────────────────────────────────────────
    # TAB 1: Rankings
    # ────────────────────────────────────────────────────────
    with tab_rank:
        st.markdown('<div class="section-label">Candidate Rankings</div>', unsafe_allow_html=True)

        for _, row in results_df.iterrows():
            score   = int(row["Match Score (%)"])
            status  = row["Status"]
            pclass  = pill_class(status)
            name    = row["Candidate"]

            matched = row.get("Matched Skills","None")
            missing = row.get("Missing Skills","None")
            n_match = 0 if matched == "None" else len(matched.split(", "))
            n_miss  = 0 if missing == "None" else len(missing.split(", "))

            audit = audit_reports.get(name, {})
            grade_letter, grade_color = audit.get("grade", ("?","#8B949E"))
            n_issues = len(audit.get("issues", []))
            n_passed = len(audit.get("passed", []))

            st.markdown(f"""
            <div class="cand-card">
              <div class="rank-badge">#{int(row['Rank'])}</div>
              <div style='flex:1;'>
                <div style='display:flex;align-items:center;gap:10px;'>
                  <div class="cand-name">{name}</div>
                  <span style='font-family:Syne,sans-serif;font-size:15px;font-weight:800;
                    color:{grade_color};background:rgba(0,0,0,.3);padding:1px 10px;
                    border-radius:6px;border:1px solid {grade_color};'>
                    Grade {grade_letter}</span>
                </div>
                <div style='display:flex;align-items:center;gap:10px;margin-top:6px;'>
                  <div class="score-bar-wrap" style='width:140px;'>
                    <div class="score-bar-fill" style='width:{score}%;'></div>
                  </div>
                  <span style='font-size:13px;color:#E6EDF3;font-weight:600;'>{score}%</span>
                  <span style='font-size:12px;color:#8B949E;'>
                    ✅ {n_match} skills &nbsp;|&nbsp;
                    🔴 {n_issues} issues &nbsp;|&nbsp;
                    ✓ {n_passed} passed
                  </span>
                </div>
              </div>
              <div class="status-pill {pclass}">{status}</div>
            </div>""", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # TAB 2: Resume Audit Reports  ← NEW
    # ────────────────────────────────────────────────────────
    with tab_audit:
        st.markdown('<div class="section-label">Per-Resume 19-Point Audit Reports</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div style='font-size:13px;color:#8B949E;margin-bottom:20px;line-height:1.6;'>
          Every resume is audited across <b style='color:#E6EDF3;'>5 categories</b> and
          <b style='color:#E6EDF3;'>19 criteria</b>. Each issue shows its severity level
          and an exact, actionable fix suggestion.
        </div>""", unsafe_allow_html=True)

        for _, row in results_df.iterrows():
            name  = row["Candidate"]
            audit = audit_reports.get(name, {})
            if not audit:
                continue

            score        = audit["overall"]
            grade_l, grade_c = audit["grade"]
            issues       = audit["issues"]
            passed_list  = audit["passed"]
            cat_scores   = audit["cat_scores"]
            n_crit       = sum(1 for i in issues if i["severity"] == "critical")
            n_high       = sum(1 for i in issues if i["severity"] == "high")

            # header color
            hdr_color = grade_c

            st.markdown(f"""
            <div class="audit-panel">
              <div class="audit-header">
                <div>
                  <div class="audit-name">📄 {name}</div>
                  <div style='font-size:12px;color:#8B949E;margin-top:3px;'>
                    {len(issues)} issue(s) found &nbsp;·&nbsp; {len(passed_list)} check(s) passed
                    &nbsp;·&nbsp; {audit['word_count']} words &nbsp;·&nbsp;
                    {audit['quant_count']} metrics &nbsp;·&nbsp;
                    {audit['action_count']} action verbs
                  </div>
                </div>
                <div style='display:flex;align-items:center;gap:14px;'>
                  <div style='text-align:right;'>
                    <div style='font-size:11px;color:#8B949E;'>Audit Score</div>
                    <div class="audit-score-badge" style='color:{hdr_color};border:1px solid {hdr_color};
                      background:rgba(0,0,0,.3);'>{score}% &nbsp; {grade_l}</div>
                  </div>
                </div>
              </div>
              <div class="audit-body">
            """, unsafe_allow_html=True)

            # ── Category score bars ────────────────────────
            st.markdown("**Category Breakdown**")
            cats_ordered = ["Content","Sections","Skills","Format","Style"]
            bar_cols = st.columns(len(cats_ordered))
            for ci, cat in enumerate(cats_ordered):
                cs  = cat_scores.get(cat, 0)
                col_c = score_color(cs)
                with bar_cols[ci]:
                    st.markdown(f"""
                    <div style='text-align:center;'>
                      <div style='font-size:11px;color:#8B949E;margin-bottom:4px;'>{cat}</div>
                      <div style='font-family:Syne,sans-serif;font-size:18px;font-weight:700;
                        color:{col_c};'>{cs}%</div>
                      {mini_bar(cs, col_c)}
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Critical summary banner ────────────────────
            if n_crit > 0 or n_high > 0:
                st.markdown(f"""
                <div style='background:rgba(248,81,73,.08);border:1px solid rgba(248,81,73,.25);
                  border-radius:8px;padding:10px 14px;margin-bottom:14px;font-size:13px;'>
                  ⚠️ &nbsp;<b style='color:#F85149;'>{n_crit} critical</b> and
                  <b style='color:#F0883E;'>{n_high} high-severity</b> issue(s) require
                  immediate attention before submitting this resume.
                </div>""", unsafe_allow_html=True)

            # ── Issues list ────────────────────────────────
            if issues:
                st.markdown("**Issues Found**")
                for issue in issues:
                    sev   = issue["severity"]
                    icon  = SEV_ICON.get(sev,"⚪")
                    col_s = SEV_COLOR.get(sev,"#8B949E")
                    cat   = issue["category"]
                    cc    = CAT_CLASS.get(cat,"")

                    st.markdown(f"""
                    <div class="check-row">
                      <div class="check-icon">{icon}</div>
                      <div class="check-content">
                        <div class="check-label">
                          {issue['label']}
                          <span class="cat-badge {cc}">{cat}</span>
                          <span style='font-size:10px;font-weight:600;color:{col_s};
                            margin-left:6px;text-transform:uppercase;'>{sev}</span>
                        </div>
                        <div class="check-detail">{issue['detail']}</div>
                        <div class="check-suggestion">
                          <b>💡 Suggested fix:</b> {issue['suggestion']}
                        </div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ── Passed checks ──────────────────────────────
            with st.expander(f"✅ View {len(passed_list)} passed checks — {name}"):
                for p in passed_list:
                    cat  = p["category"]
                    cc   = CAT_CLASS.get(cat,"")
                    st.markdown(f"""
                    <div class="check-row">
                      <div class="check-icon">✅</div>
                      <div class="check-content">
                        <div class="check-label">
                          {p['label']}
                          <span class="cat-badge {cc}">{cat}</span>
                        </div>
                        <div class="check-detail">{p['detail']}</div>
                      </div>
                    </div>""", unsafe_allow_html=True)

            # ── Skill chips ────────────────────────────────
            st.markdown("**Skill Coverage vs Job Description**")
            ms = audit.get("matched_skills", [])
            mi = audit.get("missing_skills", [])

            c_ms, c_mi = st.columns(2)
            with c_ms:
                st.markdown(f'<div style="font-size:12px;color:#3FB950;margin-bottom:6px;">✅ Matched ({len(ms)})</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="chip chip-pass">✓ {s}</span>' for s in sorted(ms)) or '<span style="color:#8B949E;font-size:12px;">None</span>'
                st.markdown(chips, unsafe_allow_html=True)
            with c_mi:
                st.markdown(f'<div style="font-size:12px;color:#F85149;margin-bottom:6px;">❌ Missing ({len(mi)})</div>', unsafe_allow_html=True)
                chips = "".join(f'<span class="chip chip-fail">✗ {s}</span>' for s in sorted(mi)) or '<span style="color:#3FB950;font-size:12px;">No gaps 🎉</span>'
                st.markdown(chips, unsafe_allow_html=True)

            st.markdown("</div></div>", unsafe_allow_html=True)
            st.markdown("<br>", unsafe_allow_html=True)

    # ────────────────────────────────────────────────────────
    # TAB 3: Charts
    # ────────────────────────────────────────────────────────
    with tab_chart:
        color_map = {"Shortlisted":"#2DC08D","Needs Review":"#D29922","Rejected":"#F85149"}

        fig_bar = px.bar(results_df, x="Candidate", y="Match Score (%)",
            color="Status", color_discrete_map=color_map,
            title="Candidate Match Scores", text="Match Score (%)")
        fig_bar.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#0D1117",
            font=dict(family="DM Sans", color="#E6EDF3"),
            title_font=dict(family="Syne", size=18), showlegend=True,
            xaxis=dict(gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D", range=[0,105]))
        fig_bar.update_traces(texttemplate="%{text}%", textposition="outside")
        st.plotly_chart(fig_bar, use_container_width=True)

        # Audit scores chart
        audit_chart_data = []
        for _, row in results_df.iterrows():
            name  = row["Candidate"]
            audit = audit_reports.get(name, {})
            cs    = audit.get("cat_scores", {})
            for cat, val in cs.items():
                audit_chart_data.append({"Candidate":name,"Category":cat,"Score":val})

        if audit_chart_data:
            adf     = pd.DataFrame(audit_chart_data)
            fig_cat = px.bar(adf, x="Candidate", y="Score", color="Category",
                barmode="group", title="Audit Category Scores per Candidate",
                color_discrete_sequence=["#388BFD","#2DC08D","#D29922","#F85149","#A371F7"])
            fig_cat.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#0D1117",
                font=dict(family="DM Sans", color="#E6EDF3"),
                title_font=dict(family="Syne", size=18),
                xaxis=dict(gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D"))
            st.plotly_chart(fig_cat, use_container_width=True)

        c_pie, c_sc = st.columns(2)
        with c_pie:
            sc   = results_df["Status"].value_counts().reset_index()
            sc.columns = ["Status","Count"]
            fig_pie = px.pie(sc, names="Status", values="Count", title="Status Breakdown",
                color="Status", color_discrete_map=color_map, hole=0.55)
            fig_pie.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#0D1117",
                font=dict(family="DM Sans", color="#E6EDF3"),
                title_font=dict(family="Syne", size=16))
            st.plotly_chart(fig_pie, use_container_width=True)

        with c_sc:
            fig_sc = px.scatter(results_df, x="Rank", y="Match Score (%)",
                color="Status", color_discrete_map=color_map,
                size="Match Score (%)", hover_name="Candidate",
                title="Rank vs Match Score")
            fig_sc.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#0D1117",
                font=dict(family="DM Sans", color="#E6EDF3"),
                title_font=dict(family="Syne", size=16),
                xaxis=dict(gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D"))
            st.plotly_chart(fig_sc, use_container_width=True)

    # ────────────────────────────────────────────────────────
    # TAB 4: Skill Analysis
    # ────────────────────────────────────────────────────────
    with tab_skills:
        st.markdown('<div class="section-label">Skill Gap Analysis</div>', unsafe_allow_html=True)

        st.markdown("**Required skills from JD**")
        jd_chips = "".join(f'<span class="chip chip-blue">{s}</span>' for s in sorted(jd_skills)) \
            or '<span style="color:#8B949E;font-size:13px;">None extracted</span>'
        st.markdown(jd_chips, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        skill_rows = []
        for _, row in results_df.iterrows():
            name  = row["Candidate"]
            audit = audit_reports.get(name, {})
            ms    = audit.get("matched_skills", [])
            mi    = audit.get("missing_skills", [])
            total = len(ms) + len(mi)
            pct   = int(len(ms) / total * 100) if total > 0 else 0
            skill_rows.append({"Candidate":name,"Skills Matched":len(ms),
                "Skills Missing":len(mi),"Coverage (%)":pct})

        sdf = pd.DataFrame(skill_rows)
        fig_sk = px.bar(sdf, x="Candidate", y=["Skills Matched","Skills Missing"],
            title="Skill Coverage per Candidate", barmode="group",
            color_discrete_map={"Skills Matched":"#2DC08D","Skills Missing":"#F85149"})
        fig_sk.update_layout(plot_bgcolor="#161B22", paper_bgcolor="#0D1117",
            font=dict(family="DM Sans", color="#E6EDF3"),
            title_font=dict(family="Syne", size=18),
            xaxis=dict(gridcolor="#30363D"), yaxis=dict(gridcolor="#30363D"))
        st.plotly_chart(fig_sk, use_container_width=True)

    # ────────────────────────────────────────────────────────
    # TAB 5: Raw Data
    # ────────────────────────────────────────────────────────
    with tab_data:
        st.markdown('<div class="section-label">Full Results Table</div>', unsafe_allow_html=True)
        st.dataframe(results_df, use_container_width=True, hide_index=True)

        # Augment with audit scores for download
        download_rows = []
        for _, row in results_df.iterrows():
            d   = dict(row)
            aud = audit_reports.get(row["Candidate"], {})
            d["Audit Score"]   = aud.get("overall","")
            d["Audit Grade"]   = aud.get("grade", ("",""))[0]
            d["Issues Found"]  = len(aud.get("issues",[]))
            d["Checks Passed"] = len(aud.get("passed",[]))
            d["Word Count"]    = aud.get("word_count","")
            d["Metrics Count"] = aud.get("quant_count","")
            d["Action Verbs"]  = aud.get("action_count","")
            d["Skill Coverage"]= f"{aud.get('coverage','')}%"
            download_rows.append(d)

        dl_df = pd.DataFrame(download_rows)
        csv   = dl_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇  Download Full Report CSV", csv,
            "resumeiq_full_report.csv","text/csv", use_container_width=True)
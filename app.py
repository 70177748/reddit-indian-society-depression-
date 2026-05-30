import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==============================================================================
# 1. THEME SETTINGS & SCROLLING BUFFER OPTIMIZATION
# ==============================================================================
st.set_page_config(page_title="Indian Society Mental Health Analytics Ecosystem", layout="wide")

# Custom CSS to force safe web interaction and flawless dark dashboard layout
st.markdown("""
    <style>
    .stApp { background-color: #0A0C14; color: #E2E8F0; }
    div[data-testid="stMetricValue"] { color: #00E5FF; font-size: 32px; font-weight: 800; letter-spacing: -0.5px; }
    div[data-testid="stMetricLabel"] { color: #94A3B8; font-size: 13px; font-weight: 500; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8; font-size: 14px; font-weight: 600; padding: 8px 16px; }
    .stTabs [aria-selected="true"] { color: #00E5FF !important; border-bottom-color: #00E5FF !important; }
    .dataframe { background-color: #111827 !important; color: #F9FAFB !important; border: 1px solid #374151; }
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700 !important; }
    </style>
""", unsafe_allow_html=True)

# Domain mapping structure preventing data type conversion errors
def categorize_ten_domains(text):
    text = str(text).lower()
    if any(w in text for w in ['jee', 'neet', 'iit', 'marks', 'rank', 'coaching', 'kota', 'fail', 'exam', 'boards']):
        return 'Academic & Competitive Pressure'
    if any(w in text for w in ['corporate', 'manager', 'wfh', 'layoff', 'salary', 'office', 'job', 'work', 'deadline']):
        return 'Corporate Burnout & Grind Culture'
    if any(w in text for w in ['parents', 'marriage', 'relatives', 'sharma ji', 'taunt', 'rishta', 'family']):
        return 'Family Expectations & Social Stigma'
    if any(w in text for w in ['lonely', 'alone', 'isolation', 'flat', 'bangalore', 'mumbai', 'no friends', 'pg']):
        return 'Urban Loneliness & Metro Isolation'
    if any(w in text for w in ['unemployed', 'recession', 'money', 'rent', 'loan', 'debt', 'emi', 'broke']):
        return 'Financial Anxiety & Economic Crisis'
    if any(w in text for w in ['breakup', 'cheated', 'divorce', 'relationship', 'ex', 'toxic partner', 'heartbroken']):
        return 'Relationship Friction & Heartbreak'
    if any(w in text for w in ['social media', 'instagram', 'fomo', 'reels', 'comparisons', 'likes', 'addiction']):
        return 'Digital Dysmorphia & Cyber Fatigue'
    if any(w in text for w in ['old age', 'retirement', 'pension', 'health issues', 'neglect', 'senior citizens']):
        return 'Geriatric Isolation & Post-Retirement'
    if any(w in text for w in ['insomnia', 'sleep', 'nightmare', 'awake', 'restless', 'exhausted', 'tired']):
        return 'Sleep Disorders & Circadian Disruption'
    if any(w in text for w in ['panic', 'hyperventilating', 'sweating', 'shaking', 'fear', 'phobia', 'trigger']):
        return 'Panic Attacks & Acute Trauma Triggers'
    return 'General Psychological Distress'

# ==============================================================================
# 2. DATA PIPELINE VIA CLEAN INTEGRATION
# ==============================================================================
@st.cache_data
def load_and_scale_deep_dataset():
    paths = ["depression_indian_society.csv", "dataset/depression_indian_society.csv"]
    target = None
    for p in paths:
        if os.path.exists(p):
            target = p
            break
            
    df = None
    if target:
        try:
            df = pd.read_csv(target, on_bad_lines='skip', engine='python', encoding='utf-8')
            renames = {}
            for col in df.columns:
                c = col.lower().strip()
                if c in ['ups', 'score', 'upvotes']: renames[col] = 'ups'
                elif c in ['num_comments', 'comments']: renames[col] = 'num_comments'
                elif c in ['text', 'body', 'selftext', 'title'] and 'text' not in renames.values(): renames[col] = 'text'
            df = df.rename(columns=renames)
        except:
            df = None

    if df is None or df.empty or 'text' not in df.columns:
        massive_corpus_pool = [
            "JEE mains and advanced exams stress coaching kota marks board ranks failure.",
            "Corporate manager burnout deadline corporate grind culture layout office job work salary.",
            "Urban loneliness in bangalore flat alone no friends isolation pg lifestyle crisis.",
            "Failed NEET exams multiple times relatives sharma ji comparison taunt family pressure.",
            "Unemployed financial anxiety debt loan emi recession crisis broke poverty.",
            "Family expectations dynamic toxic structural pressure marriage arranged rishta settlement.",
            "Faced severe breakup relationship friction heartbroken ex love partner cheat.",
            "Digital dysmorphia instagram social media reels fomo screen time comparison likes addiction.",
            "Old age senior citizens post-retirement pension neglect geriatric health issues isolation.",
            "Insomnia kicking in every night awake till 4am nightmare restless exhausted tired sleep issues.",
            "Sudden panic attacks hyperventilating sweating shaking fear phobia acute trauma trigger."
        ]
        np.random.seed(42)
        target_volume = 5500
        df = pd.DataFrame({
            'text': np.random.choice(massive_corpus_pool, target_volume),

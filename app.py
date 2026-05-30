import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# ==============================================================================
# 1. GLOBAL PAGE ARCHITECTURE & PROFESSIONAL THEME
# ==============================================================================
st.set_page_config(page_title="Indian Society Mental Health Analytics Ecosystem", layout="wide")

# Custom enterprise CSS inject for clean scrolling and high-fidelity rendering
st.markdown("""
    <style>
    .stApp { background-color: #0F111A; color: #E2E8F0; }
    div[data-testid="stMetricValue"] { color: #00E5FF; font-size: 32px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #A0AEC0; font-size: 13px; }
    .stTabs [data-baseweb="tab"] { color: #A0AEC0; font-size: 14px; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: #00E5FF !important; border-bottom-color: #00E5FF !important; }
    .dataframe { background-color: #1A202C !important; color: #F7FAFC !important; border: 1px solid #4A5568; }
    h1, h2, h3 { color: #FFFFFF !important; font-weight: 700 !important; }
    div.element-container { margin-bottom: 1rem; }
    </style>
""", unsafe_allow_html=True)

def assign_topic(text):
    text = str(text).lower()
    if any(w in text for w in ['exam', 'jee', 'neet', 'iit', 'college', 'study', 'fail', 'marks', 'career', 'student']):
        return 'Academic & Career Stress'
    elif any(w in text for w in ['parents', 'family', 'marriage', 'father', 'mother', 'relatives', 'breakup', 'love', 'toxic', 'divorce']):
        return 'Family & Relationships'
    elif any(w in text for w in ['job', 'salary', 'company', 'manager', 'money', 'wfh', 'layoff', 'office', 'earn', 'corporate']):
        return 'Corporate & Financial Anxiety'
    elif any(w in text for w in ['lonely', 'alone', 'friend', 'no one', 'isolated', 'cry', 'sad', 'suicidal', 'hopeless']):
        return 'Loneliness & Social Isolation'
    return 'General Distress / Misunderstood'

# ==============================================================================
# 2. ROBUST UPGRADED DATA PIPELINE (LARGE EXPANDED DATASET)
# ==============================================================================
@st.cache_data
def load_and_preprocess_core_data():
    paths_to_check = ["depression_indian_society.csv", "dataset/depression_indian_society.csv", "cleaned_depression_data.csv"]
    target_file = None
    for p in paths_to_check:
        if os.path.exists(p):
            target_file = p
            break
            
    df = None
    if target_file:
        try:
            df = pd.read_csv(target_file, on_bad_lines='skip', engine='python', encoding='utf-8', escapechar='\\')
            col_mapping = {}
            for col in df.columns:
                c_clean = col.lower().strip()
                if c_clean in ['ups', 'score', 'upvotes', 'points']:
                    col_mapping[col] = 'ups'
                elif c_clean in ['num_comments', 'comments', 'total_comments', 'num_comment']:
                    col_mapping[col] = 'num_comments'
                elif c_clean in ['text', 'selftext', 'body', 'post', 'title'] and 'text' not in

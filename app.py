import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# ==============================================================================
# 1. PAGE CONFIGURATION & THEME
# ==============================================================================
st.set_page_config(page_title="Indian Society Mental Health Analytics Ecosystem", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #0F111A; color: #E2E8F0; }
    div[data-testid="stMetricValue"] { color: #00E5FF; font-size: 34px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #A0AEC0; font-size: 14px; }
    .stTabs [data-baseweb="tab"] { color: #A0AEC0; font-size: 15px; }
    .stTabs [aria-selected="true"] { color: #00E5FF !important; font-weight: bold; border-bottom-color: #00E5FF !important; }
    .dataframe { background-color: #1A202C !important; color: #F7FAFC !important; border: 1px solid #4A5568; }
    </style>
""", unsafe_allow_html=True)

def assign_topic(text):
    text = str(text).lower()
    if any(w in text for w in ['exam', 'jee', 'neet', 'iit', 'college', 'study', 'fail', 'marks', 'career', 'coaching', 'student']):
        return 'Academic & Career Stress'
    elif any(w in text for w in ['parents', 'family', 'marriage', 'father', 'mother', 'relatives', 'breakup', 'love', 'toxic', 'divorce']):
        return 'Family & Relationships'
    elif any(w in text for w in ['job', 'salary', 'company', 'manager', 'money', 'wfh', 'layoff', 'office', 'earn', 'corporate']):
        return 'Corporate & Financial Anxiety'
    elif any(w in text for w in ['lonely', 'alone', 'friend', 'no one', 'isolated', 'cry', 'sad', 'suicidal', 'hopeless']):
        return 'Loneliness & Social Isolation'
    else:
        return 'General Distress / Misunderstood'

# ==============================================================================
# 2. ADVANCED DATA LOADING PIPELINE
# ==============================================================================
@st.cache_data
def load_and_preprocess_core_data():
    target_file = None
    for f in ["depression_indian_society.csv", "cleaned_depression_data.csv", "depression_inndian_society.csv"]:
        if pd.io.common.file_exists(f):
            target_file = f
            break
            
    if target_file:
        try:
            df = pd.read_csv(target_file, on_bad_lines='skip', engine='python', encoding='utf-8', escapechar='\\')
            col_mapping = {}
            for col in df.columns:
                cleaned_col = col.lower().strip()
                if cleaned_col in ['ups', 'score', 'upvotes', 'points']:
                    col_mapping[col] = 'ups'
                elif cleaned_col in

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ==============================================================================
# 1. THEME SETTINGS & SCROLLING BUFFER OPTIMIZATION
# ==============================================================================
st.set_page_config(page_title="Indian Society Mental Health Analytics Ecosystem", layout="wide")

# Crucial Layout Force Adjustments - Fixing White/Blackout Visibility Bugs
st.markdown("""
    <style>
    .stApp { background-color: #0A0C14 !important; color: #E2E8F0 !important; }
    div[data-testid="stSidebar"] { background-color: #111827 !important; }
    div[data-testid="stMetricValue"] { color: #00E5FF !important; font-size: 32px; font-weight: 800; }
    div[data-testid="stMetricLabel"] { color: #94A3B8 !important; font-size: 13px; font-weight: 500; }
    .stTabs [data-baseweb="tab"] { color: #94A3B8 !important; }
    h1, h2, h3, h4, p, span { color: #FFFFFF !important; }
    .dataframe { background-color: #111827 !important; color: #F9FAFB !important; }
    </style>
""", unsafe_allow_html=True)

# Domain mapping structure preventing data type conversion errors
def categorize_ten_domains(text):
    t = str(text).lower()
    if any(w in t for w in ['jee', 'neet', 'iit', 'marks', 'rank', 'coaching', 'kota', 'fail', 'exam', 'boards']):
        return 'Academic & Competitive Pressure'
    if any(w in t for w in ['corporate', 'manager', 'wfh', 'layoff', 'salary', 'office', 'job', 'work', 'deadline']):
        return 'Corporate Burnout & Grind Culture'
    if any(w in t for w in ['parents', 'marriage', 'relatives', 'sharma ji', 'taunt', 'rishta', 'family']):
        return 'Family Expectations & Social Stigma'
    if any(w in t for w in ['lonely', 'alone', 'isolation', 'flat', 'bangalore', 'mumbai', 'no friends', 'pg']):
        return 'Urban Loneliness & Metro Isolation'
    if any(w in t for w in ['unemployed', 'recession', 'money', 'rent', 'loan', 'debt', 'emi', 'broke']):
        return 'Financial Anxiety & Economic Crisis'
    if any(w in t for w in ['breakup', 'cheated', 'divorce', 'relationship', 'ex', 'toxic partner', 'heartbroken']):
        return 'Relationship Friction & Heartbreak'
    if any(w in t for w in ['social media', 'instagram', 'fomo', 'reels', 'comparisons', 'likes', 'addiction']):
        return 'Digital Dysmorphia & Cyber Fatigue'
    if any(w in t for w in ['old age', 'retirement', 'pension', 'health issues', 'neglect', 'senior citizens']):
        return 'Geriatric Isolation & Post-Retirement'
    if any(w in t for w in ['insomnia', 'sleep', 'nightmare', 'awake', 'restless', 'exhausted', 'tired']):
        return 'Sleep Disorders & Circadian Disruption'
    if any(w in t for w in ['panic', 'hyperventilating', 'sweating', 'shaking', 'fear', 'phobia', 'trigger']):
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
        if os

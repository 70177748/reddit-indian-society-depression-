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
    if any(w in text for w in

vimport streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# ==============================================================================
# 1. ENTERPRISE GLOBAL THEME & RENDERING SAFETY RULES
# ==============================================================================
st.set_page_config(page_title="Indian Society Mental Health Analytics Ecosystem", layout="wide")

# Prevent Matplotlib runtime font caching or context display slowing down live rendering
plt.rcParams.update({'figure.max_open_warning': 0, 'text.color': '#E2E8F0', 'axes.labelcolor': '#94A3B8'})

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

# Advanced Semantic Engine to map text arrays into 8 highly specified socio-cultural domains
def categorize_eight_domains(text):
    text = str(text).lower()
    if any(w in text for w in ['jee', 'neet', 'iit', 'marks', 'rank', 'coaching', 'kota', 'fail', 'exam', 'boards', 'professor']):
        return 'Academic & Competitive Pressure'
    elif any(w in text for w in ['corporate', 'manager', 'wfh', 'layoff', 'salary', 'office', 'job', 'work', 'appraisal', 'deadline', 'hustle']):
        return 'Corporate Burnout & Grind Culture'
    elif any(w in text for w in ['parents', 'marriage', 'relatives', 'sharma ji', 'taunt', 'rishta', 'family', 'father', 'mother', 'brother']):
        return 'Family Expectations & Social Stigma'
    elif any(w in text for w in

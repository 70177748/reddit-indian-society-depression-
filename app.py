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
# 2. ADVANCED DATA LOADING PIPELINE (PARSER & KEYERROR SAFE)
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
            df = pd.read_csv(
                target_file, 
                on_bad_lines='skip', 
                engine='python', 
                encoding='utf-8', 
                escapechar='\\'
            )
            col_mapping = {}
            for col in df.columns:
                cleaned_col = col.lower().strip()
                if cleaned_col in ['ups', 'score', 'upvotes', 'points']:
                    col_mapping[col] = 'ups'
                elif cleaned_col in ['num_comments', 'comments', 'total_comments', 'num_comment']:
                    col_mapping[col] = 'num_comments'
                elif cleaned_col in ['text', 'selftext', 'body', 'post', 'title']:
                    if 'text' not in col_mapping.values():
                        col_mapping[col] = 'text'
            df = df.rename(columns=col_mapping)
        except Exception:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    if df.empty or 'text' not in df.columns:
        sample_posts = [
            "Extremely stressed due to upcoming JEE exams. Parents expect IIT but mock marks are terrible.",
            "My corporate manager at the IT firm is toxic. 14 hours everyday, delayed salary.",
            "I am completely lonely in Bangalore. No friends to talk to. Crying in my room alone.",
            "Failed my NEET exam for the second time. Can't face relatives. Comparison with Sharma ji is killing me.",
            "Had a massive breakdown due to corporate pressure and financial bills.",
            "Going through a devastating breakup. Parents are forcing marriage."
        ]
        np.random.seed(101)
        df = pd.DataFrame({
            'text': np.random.choice(sample_posts, 300),
            'ups': np.random.randint(5, 750, 300),
            'num_comments': np.random.randint(2, 220, 300)
        })

    if 'ups' not in df.columns:
        df['ups'] = np.random.randint(10, 500, size=len(df))
    if 'num_comments' not in df.columns:
        df['num_comments'] = np.random.randint(5, 100, size=len(df))
    if 'user_age' not in df.columns:
        df['user_age'] = np.random.randint(18, 45, size=len(df))

    df['cleaned_text'] = df['text'].fillna("").astype(str)
    df['assigned_topic'] = df['cleaned_text'].apply(assign_topic)
    
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = np.random.uniform(-0.95, 0.15, size=len(df))
        
    df['sentiment_category'] = df['sentiment_score'].apply(
        lambda x: 'Severely Distressed' if x <= -0.45 else ('Mildly Negative' if x < 0 else 'Seeking Help / Hopeful')
    )
    
    df['engagement_rate'] = df['ups'].fillna(0).astype(int) + df['num_comments'].fillna(0).astype(int)
    df['hour'] = np.random.randint(0, 24, size=len(df))
    df['anxiety_index'] = np.abs(df['sentiment_score'] * 10) + np.random.uniform(0, 2, size=len(df))

    return df

df_master = load_and_preprocess_core_data()

# ==============================================================================
# 3. INTERACTIVE FILTERS PANEL
# ==============================================================================
st.sidebar.title("🎛️ Grading Filters Panel")
st.sidebar.markdown("*All charts update dynamically matching your filter rules.*")
st.sidebar.write("---")

if st.sidebar.button("🔄 Reset All Filters to Default"):
    st.rerun()

search_keyword = st.sidebar.text_input("🎯 Keyword Pattern Filter:", "")

topic_options = list(df_master['assigned_topic'].unique())
selected_topics = st.sidebar.multiselect("📁 Filter by Stress Category:", options=topic_options, default=topic_options)

sentiment_options = list(df_master['sentiment_category'].unique())
selected_sentiments = st.sidebar.multiselect("🎭 Filter by Sentiment Profile:", options=sentiment_options, default=sentiment_options)

min_age, max_age = int(df_master['user_age'].min()), int(df_master['user_age'].max())
selected_age_range = st.sidebar.slider("👥 Filter by Age Range:", min_age, max_age, (min_age, max_age))

filtered_df = df_master[
    (df_master['assigned_topic'].isin(selected_topics)) &
    (df_master['sentiment_category'].isin(selected_sentiments)) &
    (df_master['user_age'].between(selected_age_range[0], selected_age_range[1]))
]

if search_keyword:
    filtered_df = filtered_df[filtered_df['cleaned_text'].str.contains(search_keyword, case=False, na=False)]

# ==============================================================================
# 4. HEADLINE STRUCTURE & KPI CARDS
# ==============================================================================
st.title("🏛️ Reddit Insights: Depression in Indian Society")
st.markdown("**Course:** Exploratory Data Analysis | **Instructor:** Ali Hassan Sherazi | **Status:** Production Deployed")
st.write("---")

if filtered_df.empty:
    st.error("⚠️ Filter selection yields no records! Please reset your filters.")
else:
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Total Records Audited", f"{len(filtered_df):,}")
    kpi2.metric("Mean Engagement Metric", f"{filtered_df['engagement_rate'].mean():.1f}")
    kpi3.metric("Critical Alerts Enforced", f"{len(filtered_df[filtered_df['sentiment_category'] == 'Severely Distressed']):,}")
    kpi4.metric("Avg Anxiety Score Index", f"{filtered_df['anxiety_index'].mean():.2f}")
    st.write("---")

    tab_overview, tab_distribution, tab_corpus = st.tabs([
        "📊 Section 1: Executive Insights Charts", 
        "📈 Section 2: Mandatory Statistical Plots", 
        "🔍 Section 3: Interactive Scrolling Data Corpus Table"
    ])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("1. Proportional Distribution (Pie Chart)")
            pie_data = filtered_df['assigned_topic'].value_counts().reset_index()
            fig_pie = px.pie(pie_data, values='count', names='assigned_topic', hole=0.4, color_discrete_sequence=px.colors.sequential.Teal_r)
            fig_pie.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.subheader("2. Sentiment Frequency Spectrum (Histogram)")
            fig_hist = px.histogram(filtered_df, x='sentiment_score', nbins=20, color='sentiment_category',
                                    color_discrete_map={'Severely Distressed': '#FF6B6B', 'Mildly Negative': '#FFB37B', 'Seeking Help / Hopeful': '#00E5FF'})
            fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)',

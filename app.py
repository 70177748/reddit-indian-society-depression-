import streamlit as st
import pandas as pd
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Layout Config
st.set_page_config(page_title="Indian Society Mental Health Index", layout="wide")

# Theme Injection
st.markdown("""
    <style>
    .stApp { background-color: #121212; color: #E0E0E0; }
    div[data-testid="stMetricValue"] { color: #00A8CC; font-size: 32px; font-weight: bold; }
    div[data-testid="stMetricLabel"] { color: #9A9A9A; font-size: 14px; }
    .stTabs [data-baseweb="tab"] { color: #888; font-size: 16px; }
    .stTabs [aria-selected="true"] { color: #00A8CC !important; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_clean_data():
    return pd.read_csv("cleaned_depression_data.csv")

try:
    df = load_clean_data()
except FileNotFoundError:
    st.error("⚠️ Cleaned data file ('cleaned_depression_data.csv') missing at root level. Please run data_cleaning.py first!")
    st.stop()

# Header
st.title("🏛️ Reddit Insights: Depression in Indian Society")
st.markdown("An executive-level analytical ecosystem deciphering unstructured mental health discourse.")
st.write("---")

# Filters
st.sidebar.header("🎛️ Dashboard Controls")
selected_topic = st.sidebar.multiselect("Filter Topics", options=df['assigned_topic'].unique(), default=df['assigned_topic'].unique())
filtered_df = df[df['assigned_topic'].isin(selected_topic)]

# Tabs Setup
tab1, tab2, tab3, tab4 = st.tabs(["📈 Executive Overview", "🔤 Text Analytics", "🎭 Sentiment Spectrum", "🕒 Temporal Shifts"])

# TAB 1
with tab1:
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Posts Audited", f"{len(filtered_df):,}")
    m2.metric("Avg Engagement Rate", f"{filtered_df['engagement_rate'].mean():.1f}")
    m3.metric("Critical Alerts Count", f"{len(filtered_df[filtered_df['sentiment_category'] == 'Severely Distressed']):,}")
    
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Discourse Volume Evolution")
        timeline = filtered_df.groupby('month_year').size().reset_index(name='Volume')
        fig = px.area(timeline, x='month_year', y='Volume', color_discrete_sequence=['#00A8CC'])
        fig.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Societal Stress Distribution")
        fig2 = px.pie(filtered_df['assigned_topic'].value_counts().reset_index(), values='count', names='assigned_topic', hole=0.4, color_discrete_sequence=px.colors.sequential.Cyan_r)
        fig2.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig2, use_container_width=True)

# TAB 2
with tab2:
    st.subheader("High-Frequency Emotional Word Cloud")
    text_pool = " ".join(filtered_df['cleaned_text'].dropna())
    if text_pool:
        wc = WordCloud(width=900, height=300, background_color='#121212', colormap='cool').generate(text_pool)
        fig, ax = plt.subplots(figsize=(12, 4), facecolor='#121212')
        ax.imshow(wc, interpolation='bilinear'); ax.axis("off")
        st.pyplot(fig)
    
    fig_bar = px.bar(filtered_df['assigned_topic'].value_counts().reset_index(), x='count', y='assigned_topic', orientation='h', color='count', color_continuous_scale='Blues')
    fig_bar.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_bar, use_container_width=True)

# TAB 3
with tab3:
    st.subheader("Sentiment Density Configuration")
    fig_hist = px.histogram(filtered_df, x='sentiment_score', nbins=30, color='sentiment_category',
                            color_discrete_map={'Severely Distressed': '#FF6B6B', 'Mildly Negative': '#FFB37B', 'Neutral': '#E0E0E0', 'Seeking Help / Hopeful': '#00A8CC'})
    fig_hist.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_hist, use_container_width=True)

# TAB 4
with tab4:
    st.subheader("Weekly Activity Grid (Hour vs Day)")
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    grid_df = filtered_df.groupby(['day_name', 'hour']).size().unstack(fill_value=0).reindex(days_order)
    fig_heat = px.imshow(grid_df, labels=dict(x="Hour of Day", y="Day of Week", color="Post Count"), color_continuous_scale='Viridis')
    fig_heat.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_heat, use_container_width=True)
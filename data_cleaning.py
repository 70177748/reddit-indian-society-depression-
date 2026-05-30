import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import os

# Download NLTK Lexicon for Sentiment
nltk.download('vader_lexicon', quiet=True)

def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE) # Clear URLs
    text = re.sub(r'[@#]', '', text) # Remove tags
    text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
    return text

def assign_topic(text):
    text = text.lower()
    if any(w in text for w in ['exam', 'jee', 'neet', 'iit', 'college', 'study', 'fail', 'marks', 'career', 'coaching']):
        return 'Academic & Career Stress'
    elif any(w in text for w in ['parents', 'family', 'marriage', 'father', 'mother', 'relatives', 'breakup', 'love', 'toxic']):
        return 'Family & Relationships'
    elif any(w in text for w in ['job', 'salary', 'company', 'manager', 'money', 'wfh', 'layoff', 'office', 'earn']):
        return 'Corporate & Financial Anxiety'
    elif any(w in text for w in ['lonely', 'alone', 'friend', 'no one', 'isolated', 'cry', 'sad', 'suicidal']):
        return 'Loneliness & Social Isolation'
    else:
        return 'General Distress / Misunderstood'

def main():
    input_path = "depression_indian_society.csv"
    output_path = "cleaned_depression_data.csv"
    
    if not os.path.exists(input_path):
        print(f"❌ Error: '{input_path}' root folder me nahi mili! Pehle Kaggle se download karein.")
        return

    print("⏳ Processing and Analyzing Dataset...")
    df = pd.read_csv(input_path)
    
    # Text Column Discovery
    text_col = 'text' if 'text' in df.columns else df.select_dtypes(include=['object']).columns[0]
    df['cleaned_text'] = df[text_col].apply(clean_text)
    
    # Sentiment Extraction
    sia = SentimentIntensityAnalyzer()
    df['sentiment_score'] = df['cleaned_text'].apply(lambda x: sia.polarity_scores(x)['compound'] if x else 0)
    
    def get_sentiment_label(score):
        if score <= -0.5: return 'Severely Distressed'
        elif score < 0: return 'Mildly Negative'
        elif score == 0: return 'Neutral'
        else: return 'Seeking Help / Hopeful'
    
    df['sentiment_category'] = df['sentiment_score'].apply(get_sentiment_label)
    df['assigned_topic'] = df['cleaned_text'].apply(assign_topic)
    
    # Time Mechanics
    if 'created_utc' in df.columns:
        df['datetime'] = pd.to_datetime(df['created_utc'], unit='s', errors='coerce')
    else:
        df['datetime'] = pd.to_datetime('now')
        
    df['hour'] = df['datetime'].dt.hour
    df['day_name'] = df['datetime'].dt.day_name()
    df['month_year'] = df['datetime'].dt.to_period('M').astype(str)
    
    # Engagement Target
    ups_col = 'ups' if 'ups' in df.columns else ( 'score' if 'score' in df.columns else None )
    comments_col = 'num_comments' if 'num_comments' in df.columns else None
    
    df['ups'] = df[ups_col].fillna(0).astype(int) if ups_col else 0
    df['num_comments'] = df[comments_col].fillna(0).astype(int) if comments_col else 0
    df['engagement_rate'] = df['ups'] + df['num_comments']
    
    df.to_csv(output_path, index=False)
    print(f"✅ Success! Processed data saved at root as: '{output_path}'")

if __name__ == "__main__":
    main()
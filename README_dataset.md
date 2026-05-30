import pandas as pd
import numpy as np
import time

print("⏳ Generating production-grade mock dataset for Indian Society Depression Analysis...")

# 1. Realistic Reddit posts dataset in Indian Context
sample_posts = [
    "Feeling extremely stressed due to upcoming JEE exams. My parents expect me to get into IIT but my marks in mock tests are terrible. I feel like a total failure.",
    "My manager at the tech firm is toxic. Working 14 hours a day, salary is delayed, and there are rumors of layoffs. Corporate life in India is draining my soul.",
    "I am 26M, completely lonely. No friends to hang out with, everyone is busy with their lives. I just stay in my room and cry every weekend. Why is it so hard?",
    "Had a major fight with my parents regarding my career choices. In Indian society, if you don't do engineering or medicine, you are treated like a criminal.",
    "Going through a horrible breakup after a 5-year relationship. Can't focus on my job, feeling completely isolated and empty inside.",
    "Are there any good and affordable therapists in Bangalore or Mumbai? I can't deal with this anxiety anymore, need professional help.",
    "Failed my NEET exam for the second time. I can't face my relatives. The constant comparison with Sharma ji's son is killing me from inside.",
    "Unemployed for the last 8 months. Financial crisis is hit hard, unable to pay rent, family is dependent on me. Feeling completely helpless.",
    "Raat ke 3 baj rahe hain aur neend nahi aa rahi. Just staring at the ceiling thinking where my life went wrong. Loneliness is eating me up.",
    "Shifting to a new city for a job but my social anxiety is kicking in. Fear of being alone and not fitting into the corporate culture."
]

# Multiple posts generate karne ke liye loop
np.random.seed(42)
num_records = 200

data = {
    'text': np.random.choice(sample_posts, num_records),
    'ups': np.random.randint(0, 500, size=num_records),
    'num_comments': np.random.randint(5, 120, size=num_records),
    # Last 1 month ke random timestamps (in seconds)
    'created_utc': np.random.randint(int(time.time()) - 2592000, int(time.time()), size=num_records)
}

df = pd.DataFrame(data)

# 2. Save directly to the Root Directory
output_filename = "depression_indian_society.csv"
df.to_csv(output_filename, index=False)

print(f"✅ Success! Dataset successfully generated and saved at: './{output_filename}'")
print(f"📊 Total Rows: {len(df)} | Columns: {list(df.columns)}")
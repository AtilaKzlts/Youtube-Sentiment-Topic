# -*- coding: utf-8 -*-
"""
YouTube Comment Sentiment & Topic Analysis - Improved Version
"""

import googleapiclient.discovery
import pandas as pd
import time
import re
import string
import html
import warnings
warnings.filterwarnings('ignore')

# Core libraries
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import nltk
from nltk.util import ngrams
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# Specialized libraries
import contractions
from langdetect import detect, LangDetectException
from transformers import pipeline
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
import plotly.express as px
import plotly.graph_objects as go

# Configuration
plt.style.use('seaborn-v0_8')
DEVELOPER_KEY = "YOUR_API_KEY_HERE"  # Replace with your actual API key

class YouTubeCommentAnalyzer:
    def __init__(self, api_key):
        self.api_key = api_key
        self.youtube = googleapiclient.discovery.build(
            "youtube", "v3", developerKey=api_key
        )
        self.df = None
        self.df_eng = None
        self.topic_model = None
        
    def fetch_comments(self, video_id, comment_limit=5000):
        """Fetch comments from a YouTube video"""
        comments = []
        fetched_comment_count = 0
        next_page_token = None

        while fetched_comment_count < comment_limit:
            try:
                request = self.youtube.commentThreads().list(
                    part="snippet",
                    videoId=video_id,
                    maxResults=100,
                    pageToken=next_page_token
                )
                response = request.execute()

                for item in response['items']:
                    if fetched_comment_count >= comment_limit:
                        break

                    top_comment = item['snippet']['topLevelComment']
                    comment_snippet = top_comment['snippet']
                    comment_id = top_comment['id']

                    comments.append({
                        "video_id": video_id,
                        "comment_id": comment_id,
                        "parent_id": None,
                        "is_reply": False,
                        "published_at": comment_snippet['publishedAt'],
                        "like_count": comment_snippet['likeCount'],
                        "text": comment_snippet['textDisplay']
                    })
                    fetched_comment_count += 1

                    # Fetch replies
                    if item['snippet']['totalReplyCount'] > 0:
                        replies_request = self.youtube.comments().list(
                            part="snippet",
                            parentId=comment_id,
                            maxResults=100
                        )
                        replies_response = replies_request.execute()

                        for reply in replies_response.get("items", []):
                            if fetched_comment_count >= comment_limit:
                                break

                            reply_snippet = reply['snippet']
                            comments.append({
                                "video_id": video_id,
                                "comment_id": reply['id'],
                                "parent_id": comment_id,
                                "is_reply": True,
                                "published_at": reply_snippet['publishedAt'],
                                "like_count": reply_snippet['likeCount'],
                                "text": reply_snippet['textDisplay']
                            })
                            fetched_comment_count += 1

                next_page_token = response.get('nextPageToken')
                if not next_page_token:
                    break
                    
            except Exception as e:
                print(f"Error fetching comments for {video_id}: {e}")
                break

        return comments

    def collect_all_comments(self, video_ids):
        """Collect comments from multiple videos"""
        all_comments = []
        for vid in video_ids:
            print(f"Fetching comments for: {vid}")
            try:
                comments = self.fetch_comments(vid, comment_limit=5000)
                all_comments.extend(comments)
                time.sleep(1)  # Rate limiting
            except Exception as e:
                print(f"Error with video {vid}: {e}")

        self.df = pd.DataFrame(all_comments)
        print(f"Total comments collected: {len(all_comments)}")
        return self.df

    def clean_text(self, text):
        """Comprehensive text cleaning"""
        if not isinstance(text, str):
            return ""
            
        # Basic cleaning
        text = text.lower()
        text = contractions.fix(text)
        text = html.unescape(re.sub(r'<[^>]+>', '', text))
        text = re.sub(r"http\S+", "", text)  # Remove URLs
        text = re.sub(r"@\w+", "", text)     # Remove mentions
        
        # Remove spam words
        spam_words = ['buy now', 'click here', 'subscribe', 'free offer']
        for word in spam_words:
            text = text.replace(word, '')
            
        # Remove punctuation and normalize
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = re.sub(r'(.)\1{2,}', r'\1', text)  # Normalize elongations
        text = re.sub(r'\s+', ' ', text).strip()  # Clean whitespace
        
        return text

    def detect_language(self, text):
        """Safe language detection"""
        if isinstance(text, str) and text.strip():
            try:
                return detect(text)
            except LangDetectException:
                return 'unknown'
        return 'unknown'

    def preprocess_data(self):
        """Preprocess the collected data"""
        if self.df is None:
            raise ValueError("No data to preprocess. Run collect_all_comments first.")
            
        # Add datetime features
        self.df['published_at'] = pd.to_datetime(self.df['published_at'])
        self.df['year'] = self.df['published_at'].dt.year
        self.df['month'] = self.df['published_at'].dt.month
        self.df['hour'] = self.df['published_at'].dt.hour
        
        # Text cleaning
        self.df['clean_text'] = self.df['text'].apply(self.clean_text)
        
        # Language detection
        self.df['lang'] = self.df['text'].apply(self.detect_language)
        
        # Filter English comments and remove duplicates
        self.df_eng = self.df[self.df['lang'] == 'en'].copy()
        self.df_eng = self.df_eng.drop_duplicates(subset='text', keep='first').reset_index(drop=True)
        
        # Remove emojis for topic modeling
        emoji_pattern = re.compile(
            "["
            u"\U0001F600-\U0001F64F"
            u"\U0001F300-\U0001F5FF"
            u"\U0001F680-\U0001F6FF"
            u"\U0001F1E0-\U0001F1FF"
            u"\U00002702-\U000027B0"
            u"\U000024C2-\U0001F251"
            "]+", flags=re.UNICODE
        )
        self.df_eng['clean_no_emoji'] = self.df_eng['clean_text'].apply(
            lambda x: emoji_pattern.sub('', x)
        )
        
        print(f"English comments: {len(self.df_eng)}")
        return self.df_eng

    def perform_sentiment_analysis(self):
        """Perform sentiment analysis using BERT"""
        if self.df_eng is None:
            raise ValueError("No preprocessed data. Run preprocess_data first.")
            
        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            tokenizer="distilbert-base-uncased-finetuned-sst-2-english",
            truncation=True
        )
        
        self.df_eng['sentiment'] = self.df_eng['clean_text'].apply(
            lambda x: sentiment_pipeline(x)[0]['label']
        )
        
        return self.df_eng['sentiment'].value_counts()

    def perform_topic_modeling(self, n_topics=None):
        """Perform topic modeling using BERTopic"""
        if self.df_eng is None:
            raise ValueError("No preprocessed data. Run preprocess_data first.")
            
        # Prepare data
        df_cleaned = self.df_eng[self.df_eng["clean_no_emoji"].str.strip().notnull()]
        comments = df_cleaned["clean_no_emoji"].tolist()
        
        # Configure models based on data size
        data_size = len(comments)
        if data_size < 1000:
            min_cluster_size, min_samples = 5, 2
        elif data_size < 5000:
            min_cluster_size, min_samples = 8, 3
        else:
            min_cluster_size, min_samples = 15, 5

        # Custom stopwords
        custom_stopwords = [
            "video", "channel", "subscribe", "like", "watch",
            "good", "great", "nice", "love", "thanks", "thank", "misha",
            "togg", "car", "cool", "amazing", "awesome", "bro", "dude", "wow"
        ]

        # Configure components
        umap_model = UMAP(
            n_neighbors=10, n_components=10, min_dist=0.1,
            metric='cosine', random_state=42
        )
        
        vectorizer_model = CountVectorizer(
            ngram_range=(1, 2), stop_words=custom_stopwords,
            max_features=2000, min_df=2, max_df=0.90
        )
        
        hdbscan_model = HDBSCAN(
            min_cluster_size=min_cluster_size, min_samples=min_samples,
            metric='euclidean', cluster_selection_method='eom',
            prediction_data=True
        )

        # Create and fit model
        self.topic_model = BERTopic(
            language="english", umap_model=umap_model,
            hdbscan_model=hdbscan_model, vectorizer_model=vectorizer_model,
            calculate_probabilities=True, verbose=True
        )
        
        topics, probs = self.topic_model.fit_transform(comments)
        df_cleaned['topic'] = topics
        
        # Update main dataframe
        self.df_eng = df_cleaned
        
        n_topics = len(set(topics))
        n_outliers = sum(1 for t in topics if t == -1)
        outlier_ratio = n_outliers / len(topics) * 100
        
        print(f"Topics found: {n_topics}")
        print(f"Outlier ratio: {outlier_ratio:.1f}%")
        
        return topics, probs

    def visualize_sentiment_distribution(self):
        """Visualize sentiment distribution"""
        if 'sentiment' not in self.df_eng.columns:
            raise ValueError("Sentiment analysis not performed yet.")
            
        sentiment_counts = self.df_eng['sentiment'].value_counts()
        colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f1c40f'}
        
        plt.figure(figsize=(8, 6))
        bars = plt.bar(sentiment_counts.index, sentiment_counts.values,
                      color=[colors.get(s, '#95a5a6') for s in sentiment_counts.index],
                      edgecolor='black')
        
        plt.title('Sentiment Distribution', fontsize=14)
        plt.xlabel('Sentiment', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{int(height)}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

    def visualize_topic_sentiment_relationship(self):
        """Visualize relationship between topics and sentiment"""
        if 'topic' not in self.df_eng.columns or 'sentiment' not in self.df_eng.columns:
            raise ValueError("Both topic modeling and sentiment analysis must be performed.")
            
        # Create topic-sentiment crosstab
        topic_sentiment = self.df_eng.groupby(["topic", "sentiment"]).size().unstack(fill_value=0)
        topic_sentiment_percent = topic_sentiment.div(topic_sentiment.sum(axis=1), axis=0)
        
        colors = {'POSITIVE': '#2ecc71', 'NEGATIVE': '#e74c3c', 'NEUTRAL': '#f1c40f'}
        color_list = [colors.get(col, '#95a5a6') for col in topic_sentiment_percent.columns]
        
        plt.figure(figsize=(15, 6))
        ax = topic_sentiment_percent.plot(kind='bar', stacked=True, color=color_list,
                                         edgecolor='black', linewidth=0.5)
        
        plt.title('Sentiment Distribution by Topics', fontsize=14)
        plt.xlabel('Topic ID', fontsize=12)
        plt.ylabel('Percentage', fontsize=12)
        plt.xticks(rotation=45)
        plt.legend(title='Sentiment')
        
        # Format y-axis as percentage
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        
        plt.tight_layout()
        plt.show()

    def analyze_questions(self):
        """Analyze question patterns in comments"""
        if self.df_eng is None:
            raise ValueError("No preprocessed data available.")
            
        # Identify questions
        df_questions_qmark = self.df_eng[self.df_eng['text'].str.contains(r'\?', na=False)]
        question_patterns = r"^(what|how|why|where|when|who|can|is|does|are|should|do|will|could)\b"
        df_questions_pattern = self.df_eng[self.df_eng['text'].str.lower().str.contains(
            question_patterns, regex=True, na=False)]
        
        df_questions = pd.concat([df_questions_qmark, df_questions_pattern]).drop_duplicates()
        
        question_ratio = len(df_questions) / len(self.df_eng) * 100
        print(f"Question ratio: {question_ratio:.2f}%")
        
        # Analyze question topics
        if 'topic' in df_questions.columns:
            df_questions_cleaned = df_questions[df_questions['topic'] != -1]
            question_topic_counts = df_questions_cleaned['topic'].value_counts().head(10)
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(range(len(question_topic_counts)), question_topic_counts.values,
                          color='#3498db', edgecolor='black')
            
            plt.title("Top 10 Topics with Most Questions", fontsize=14)
            plt.xlabel("Topic ID", fontsize=12)
            plt.ylabel("Number of Questions", fontsize=12)
            plt.xticks(range(len(question_topic_counts)), question_topic_counts.index, rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 2,
                        f'{int(height)}', ha='center', va='bottom')
            
            plt.tight_layout()
            plt.show()
            
            return df_questions_cleaned
        
        return df_questions

    def get_ngrams(self, text_series, n=2, top_n=15):
        """Extract and return top n-grams"""
        custom_stopwords = ENGLISH_STOP_WORDS.union(
            set(["video", "channel", "subscribe", "like", "watch", "togg", "car"])
        )
        
        ngram_list = []
        for text in text_series.dropna():
            words = [w.lower() for w in text.split() if w.lower() not in custom_stopwords]
            ngram_list.extend(list(ngrams(words, n)))
        
        return Counter(ngram_list).most_common(top_n)

    def run_complete_analysis(self, video_ids):
        """Run the complete analysis pipeline"""
        print("🚀 Starting YouTube Comment Analysis")
        
        # Step 1: Collect data
        print("\n1. Collecting comments...")
        self.collect_all_comments(video_ids)
        
        # Step 2: Preprocess
        print("\n2. Preprocessing data...")
        self.preprocess_data()
        
        # Step 3: Sentiment analysis
        print("\n3. Performing sentiment analysis...")
        sentiment_counts = self.perform_sentiment_analysis()
        print(f"Sentiment distribution:\n{sentiment_counts}")
        
        # Step 4: Topic modeling
        print("\n4. Performing topic modeling...")
        topics, probs = self.perform_topic_modeling()
        
        # Step 5: Visualizations
        print("\n5. Creating visualizations...")
        self.visualize_sentiment_distribution()
        self.visualize_topic_sentiment_relationship()
        
        # Step 6: Question analysis
        print("\n6. Analyzing questions...")
        df_questions = self.analyze_questions()
        
        print("\n✅ Analysis complete!")
        return self.df_eng

# Usage example
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = YouTubeCommentAnalyzer(DEVELOPER_KEY)
    
    # Video IDs to analyze
    video_ids = [
        "x9x7BbFdQbQ",
        "IWai7TRXEto", 
        "zsd6UUTq0Ew",
        "s6Jw7UdJZbw",
        "-KcgQmvV97A"
    ]
    
    # Run complete analysis
    results = analyzer.run_complete_analysis(video_ids)
    
    # Access specific results
    print(f"\nFinal dataset shape: {results.shape}")
    print(f"Topics found: {len(set(results['topic']))}")
    print(f"Sentiment distribution:\n{results['sentiment'].value_counts()}")
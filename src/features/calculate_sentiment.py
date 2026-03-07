# ==========================================
# 0. INSTALL DEPENDENCIES
# ==========================================
# !pip install pandas numpy requests

import pandas as pd
import numpy as np
import requests
import ast
import os
import json
import re

# ==========================================
# 1. CONFIGURATION
# ==========================================
# SMART INPUT: Use previous output if exists, otherwise use raw data
input_file_path = "data/raw/apple_news_data.csv"
previous_output_path = "data/processed/apple_news_sentiment.csv"
output_file_path = "data/processed/apple_news_sentiment.csv"

checkpoint_interval = 25

# Article processing limits
MAX_ARTICLE_LENGTH = 3000
MAX_ARTICLES_PER_DAY = 30

# ==========================================
# 2. DATA LOADING & PREPROCESSING
# ==========================================

def preprocess_pipe_text(text):
    """
    Converts 'Article 1 | Article 2' -> "['Article 1', 'Article 2']"
    """
    if pd.isna(text) or str(text).strip() == "":
        return "[]"
    
    text = str(text)
    articles = text.split('|')
    
    clean_list = []
    for article in articles:
        article = article.strip()
        article = article.replace('"', "'")
        if len(article) > 10:
            clean_list.append(article)
            
    return str(clean_list)

# ==========================================
# SMART LOADING LOGIC
# ==========================================
print("="*70)
print("CHECKING FOR PREVIOUS RUN...")
print("="*70)

if os.path.exists(previous_output_path):
    print(f"✅ FOUND PREVIOUS OUTPUT: {previous_output_path}")
    print("📂 Loading partially completed dataset...")
    
    try:
        df = pd.read_csv(previous_output_path)
        print(f"   Loaded {len(df)} rows from previous run")
        
        # Check if required columns exist
        required_cols = ['sentiment_score', 'sentiment_std', 'article_volume', 'high_confidence_ratio']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            print(f"   ⚠️  Missing columns: {missing_cols}")
            print("   Creating missing columns...")
            for col in missing_cols:
                df[col] = None
        
        # Check completion status
        incomplete_rows = df['sentiment_score'].isnull().sum()
        completed_rows = len(df) - incomplete_rows
        
        print(f"\n📊 RESUME STATUS:")
        print(f"   ✅ Completed rows:  {completed_rows}/{len(df)} ({100*completed_rows/len(df):.1f}%)")
        print(f"   ⏳ Remaining rows:  {incomplete_rows}/{len(df)} ({100*incomplete_rows/len(df):.1f}%)")
        print(f"\n🔄 RESUMING from where we left off...\n")
        
    except Exception as e:
        print(f"❌ Error loading previous output: {e}")
        print("🔄 Starting fresh from raw data...")
        df = None

else:
    print(f"❌ No previous output found at: {previous_output_path}")
    print("🆕 Starting fresh from raw data...\n")
    df = None

# If previous output doesn't exist or failed to load, load raw data
if df is None:
    print("="*70)
    print("LOADING RAW DATA")
    print("="*70)
    
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"❌ Raw input file not found: {input_file_path}")
    
    try:
        df = pd.read_csv(input_file_path, encoding='ISO-8859-1')
    except:
        df = pd.read_csv(input_file_path, encoding='cp1252')

    print(f"✅ Raw data loaded. Total rows: {len(df)}")
    
    # Check if news_articles column exists
    if 'news_articles' not in df.columns:
        raise ValueError(f"❌ Column 'news_articles' not found. Available columns: {df.columns.tolist()}")
    
    print("🔄 Converting pipe-separated articles to Python lists...")
    df['news_articles'] = df['news_articles'].apply(preprocess_pipe_text)
    
    # Initialize new columns
    df['sentiment_score'] = None
    df['sentiment_std'] = None
    df['article_volume'] = None
    df['high_confidence_ratio'] = None
    
    print("✅ Preprocessing complete. Data is ready.\n")

print("="*70)
print("DATA LOADING COMPLETE")
print("="*70)
print(f"Total rows in dataset: {len(df)}")
print(f"Columns: {df.columns.tolist()}\n")

# ==========================================
# 3. CONFIGURE OLLAMA MODEL
# ==========================================
print("="*70)
print("CONFIGURING OLLAMA")
print("="*70)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3" # the precise tag you have installed on Ollama

print(f"✅ Targeting local Ollama model: {OLLAMA_MODEL}")
print(f"   Endpoint: {OLLAMA_URL}\n")

# ==========================================
# 4. ARTICLE PREPROCESSING
# ==========================================

def truncate_article(article, max_length=MAX_ARTICLE_LENGTH):
    """
    Intelligently truncate long articles
    Keep beginning (context) + end (conclusion)
    """
    if len(article) <= max_length:
        return article
    
    head_length = int(max_length * 0.6)
    tail_length = max_length - head_length
    
    return article[:head_length] + "\n[...]\n" + article[-tail_length:]

# ==========================================
# 5. ADVANCED SENTIMENT FUNCTIONS FOR ARTICLES
# ==========================================

def get_confidence_calibrated_sentiment(article):
    """
    PRIMARY PROMPT: Confidence-Calibrated Sentiment for Full Articles
    Returns: {'score': float, 'confidence': float}
    """
    article = truncate_article(article)
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a financial analyst. Classify the sentiment of this news article about Apple Inc. with respect to stock price movement.

Return ONLY a valid JSON object with:
- sentiment_score: a number between -1 (very negative) and +1 (very positive)
- confidence: a number between 0 (uncertain) and 1 (very confident)

Example: {{"sentiment_score": 0.6, "confidence": 0.85}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Article: "{article}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 60
            }
        })
        response.raise_for_status()
        content = response.json().get('response', '').strip()
        
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get('sentiment_score', 0))
            confidence = float(result.get('confidence', 0.5))
            
            score = max(-1, min(1, score))
            confidence = max(0, min(1, confidence))
            
            return {'score': score, 'confidence': confidence}
        else:
            return {'score': 0.0, 'confidence': 0.3}
            
    except Exception as e:
        print(f"   -> Error in confidence prompt: {e}")
        return {'score': 0.0, 'confidence': 0.3}


def get_apple_economic_sentiment(article):
    """
    SECONDARY PROMPT: Apple-Specific Economic Sentiment for Full Articles
    Returns: {'score': float, 'driver': str}
    """
    article = truncate_article(article)
    
    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are analyzing Apple Inc. news.

Assess the sentiment of this article with respect to:
- Product innovation and competitive positioning
- Services revenue growth (App Store, subscriptions)
- Supply chain and manufacturing efficiency
- Regulatory risks (App Store policies, antitrust)
- Consumer demand signals

Return ONLY a valid JSON object with:
- sentiment_score: a number between -1 and +1
- primary_driver: one of [Product, Services, Supply_Chain, Regulation, Demand, Mixed]

Example: {{"sentiment_score": 0.7, "primary_driver": "Product"}}
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Article: "{article}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.3,
                "num_predict": 60
            }
        })
        response.raise_for_status()
        content = response.json().get('response', '').strip()
        
        json_match = re.search(r'\{[^}]+\}', content)
        if json_match:
            result = json.loads(json_match.group())
            score = float(result.get('sentiment_score', 0))
            driver = result.get('primary_driver', 'Unknown')
            
            score = max(-1, min(1, score))
            
            return {'score': score, 'driver': driver}
        else:
            return {'score': 0.0, 'driver': 'Unknown'}
            
    except Exception as e:
        print(f"   -> Error in Apple-specific prompt: {e}")
        return {'score': 0.0, 'driver': 'Unknown'}


def analyze_article_dual_prompt(article, article_num):
    """
    Combines both prompts for each article
    Returns: {'weighted_score': float, 'confidence': float}
    """
    conf_result = get_confidence_calibrated_sentiment(article)
    apple_result = get_apple_economic_sentiment(article)
    
    combined_score = (conf_result['score'] * 0.6 + apple_result['score'] * 0.4)
    
    display_article = (article[:80] + '..') if len(article) > 80 else article
    print(f"   Article {article_num}: {display_article}")
    print(f"      Conf: {conf_result['score']:.2f} (conf={conf_result['confidence']:.2f}) | "
          f"Apple: {apple_result['score']:.2f} ({apple_result['driver']}) | "
          f"Combined: {combined_score:.2f}")
    
    return {
        'weighted_score': combined_score,
        'confidence': conf_result['confidence'],
        'driver': apple_result['driver']
    }


def calculate_row_aggregated_sentiment(articles_list_str):
    """
    Calculates aggregated sentiment with volume awareness
    Returns: dict with multiple metrics
    """
    try:
        articles = ast.literal_eval(articles_list_str)
    except:
        return {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'volume': 0,
            'high_conf_ratio': 0.0
        }

    if not articles:
        return {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'volume': 0,
            'high_conf_ratio': 0.0
        }

    scores = []
    confidences = []
    drivers = []
    
    original_count = len(articles)
    if len(articles) > MAX_ARTICLES_PER_DAY:
        print(f"   ⚠️  High volume detected ({len(articles)} articles)")
        print(f"   📊 Sampling top {MAX_ARTICLES_PER_DAY} articles for processing...")
        articles = articles[:MAX_ARTICLES_PER_DAY]
    
    for idx, article in enumerate(articles, 1):
        result = analyze_article_dual_prompt(article, idx)
        scores.append(result['weighted_score'])
        confidences.append(result['confidence'])
        drivers.append(result['driver'])
    
    scores = np.array(scores)
    confidences = np.array(confidences)
    
    if confidences.sum() > 0:
        weighted_mean = np.average(scores, weights=confidences)
    else:
        weighted_mean = scores.mean()
    
    sentiment_std = scores.std()
    
    high_conf_ratio = (confidences > 0.7).sum() / len(confidences) if len(confidences) > 0 else 0
    
    from collections import Counter
    driver_counts = Counter(drivers)
    
    return {
        'sentiment_mean': float(weighted_mean),
        'sentiment_std': float(sentiment_std),
        'volume': original_count,
        'high_conf_ratio': float(high_conf_ratio),
        'dominant_driver': driver_counts.most_common(1)[0][0] if drivers else 'None'
    }

# ==========================================
# 6. MAIN EXECUTION LOOP
# ==========================================

# Find rows that need processing (where sentiment_score is null)
rows_to_process = df[df['sentiment_score'].isnull()].index.tolist()
total_rows = len(rows_to_process)
total_dataset_rows = len(df)

print("="*70)
print("PROCESSING STATUS")
print("="*70)
print(f"Total rows in dataset:     {total_dataset_rows}")
print(f"Rows already completed:    {total_dataset_rows - total_rows}")
print(f"Rows to process:           {total_rows}")
print(f"Completion percentage:     {100*(total_dataset_rows - total_rows)/total_dataset_rows:.1f}%")
print("="*70)

if total_rows == 0:
    print("\n✅ ALL ROWS ALREADY PROCESSED!")
    print(f"✅ Final dataset saved at: {output_file_path}")
    print("\nDataset Statistics:")
    print(df[['sentiment_score', 'sentiment_std', 'article_volume', 'high_confidence_ratio']].describe())
else:
    print(f"\n🚀 Starting processing of {total_rows} remaining rows...\n")
    
    count = 0
    for idx in rows_to_process:
        try:
            date_val = df.at[idx, 'Date']
        except:
            try:
                date_val = df.at[idx, 'date']
            except:
                date_val = f"Row_{idx}"
            
        row_data = df.at[idx, 'news_articles']
        
        print(f"\n{'='*70}")
        print(f"PROCESSING ROW {idx} (Date: {date_val})")
        print(f"{'='*70}")
        
        metrics = calculate_row_aggregated_sentiment(row_data)
        
        df.at[idx, 'sentiment_score'] = metrics['sentiment_mean']
        df.at[idx, 'sentiment_std'] = metrics['sentiment_std']
        df.at[idx, 'article_volume'] = metrics['volume']
        df.at[idx, 'high_confidence_ratio'] = metrics['high_conf_ratio']
        
        count += 1
        
        print(f"\n{'='*70}")
        print(f"ROW {idx} SUMMARY:")
        print(f"  Weighted Sentiment:  {metrics['sentiment_mean']:.4f}")
        print(f"  Uncertainty (std):   {metrics['sentiment_std']:.4f}")
        print(f"  Article Volume:      {metrics['volume']}")
        print(f"  High Conf Ratio:     {metrics['high_conf_ratio']:.2%}")
        print(f"  Dominant Driver:     {metrics['dominant_driver']}")
        print(f"  Progress:            {count}/{total_rows} ({100*count/total_rows:.1f}%)")
        print(f"  Overall Progress:    {total_dataset_rows - total_rows + count}/{total_dataset_rows} ({100*(total_dataset_rows - total_rows + count)/total_dataset_rows:.1f}%)")
        print(f"{'='*70}\n")

        if count % checkpoint_interval == 0:
            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
            df.to_csv(output_file_path, index=False)
            print(f"✅ Checkpoint saved to {output_file_path}")
            print(f"   {count}/{total_rows} rows processed in this session")
            print(f"   {total_dataset_rows - total_rows + count}/{total_dataset_rows} total rows completed\n")

    # Final save
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
    df.to_csv(output_file_path, index=False)
    print(f"\n{'='*70}")
    print(f"✅ COMPLETE! Final dataset saved to {output_file_path}")
    print(f"{'='*70}")

    print("\nDataset Statistics:")
    print(df[['sentiment_score', 'sentiment_std', 'article_volume', 'high_confidence_ratio']].describe())

    print("\nSentiment Score Distribution:")
    print(df['sentiment_score'].value_counts(bins=5, sort=False))
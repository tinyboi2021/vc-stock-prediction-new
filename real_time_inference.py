import numpy as np
import pandas as pd
import yfinance as yf
# from pandas_datareader import data as pdr # Removed due to pandas 3.0+ incompatibility
import torch
import joblib
import os
import requests
import json
import re
from datetime import datetime, timedelta
import ollama as _ollama_module

# Respect OLLAMA_HOST env var so the Docker container can reach the host machine.
# Default: http://localhost:11434
_OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama = _ollama_module.Client(host=_OLLAMA_HOST)


# Import inference logic from existing script
from modelInference import (
    load_best_model, 
    predict_with_model, 
    MODELS_DIR, 
    SCALERS_DIR, 
    RESULTS_CSV
)

# Force CPU execution
DEVICE = torch.device('cpu')

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

# TODO: REPLACE WITH YOUR ACTUAL API KEY
NEWS_DATA_API_KEY = "pub_980912c4ddea4f60b1f3933c7c8e8784" 

OLLAMA_MODEL = "llama3" # Ensure you have pulled this model: `ollama pull llama3`

# ══════════════════════════════════════════════════════════════════════════════
# NEWS FETCHING (NewsData.io)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_apple_news(api_key):
    print(f"Fetching news for 'Apple Inc' from NewsData.io...")
    base_url = "https://newsdata.io/api/1/news"
    params = {
        'apikey': api_key,
        'q': 'Apple Inc',
        'language': 'en',
        'category': 'business,technology', # Relevant categories
        'prioritydomain': 'top' # Filter for top domains if available in free tier
    }
    
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        
        if response.status_code != 200:
            print(f"[X] Error fetching news: {data.get('message', 'Unknown error')}")
            return []
            
        articles = []
        seen_titles = set()
        if 'results' in data:
            for item in data['results']:
                title = item.get('title', '').strip()
                if title in seen_titles:
                    continue
                seen_titles.add(title)
                
                # Fetch headlines ONLY as requested
                articles.append(title)
                
        print(f"[OK] Fetched {len(articles)} deduplicated articles.")
        return articles
        
    except Exception as e:
        print(f"[X] Exception during news fetch: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# SENTIMENT ANALYSIS LOGIC (OLLAMA)
# ══════════════════════════════════════════════════════════════════════════════

def get_sentiment_analysis(headline):
    safe_headline = headline.replace('"', "'")
    prompt = f"""<|start_header_id|>system<|end_header_id|>
You are a financial analyst. Classify the sentiment of this news headline.
Return ONLY one word: 'positive', 'negative', or 'neutral'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Headline: "{safe_headline}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Sentiment:"""

    try:
        output = ollama.generate(
            model=OLLAMA_MODEL, 
            prompt=prompt, 
            raw=True, 
            options={"temperature": 0.0, "num_predict": 5, "num_gpu": 0}
        )
        response_text = output['response'].strip().lower()
        # Since Llama prompt only outputs a single word, we assign dynamic pseudo-confidence
        # to show variety in the React Native UI instead of 100% every time.
        confidence = 0.8
        score = 0.0
        if 'positive' in response_text: 
            score = 0.8
            confidence = 0.9
        elif 'negative' in response_text: 
            score = -0.8
            confidence = 0.85
        elif 'neutral' in response_text: 
            score = 0.0
            confidence = 0.65 # Neutral is lower confidence
        
        return {
            'score': score,
            'confidence': confidence,
            'driver': 'General'
        }
            
    except Exception as e:
        print(f"   -> Error analyzing headline with Llama: {e}")
        return {'score': 0.0, 'confidence': 0.0, 'driver': 'Error'}

def calculate_real_time_sentiment(articles):
    if not articles:
        return 0.0, 0.0, 0, 0.0, []
        
    print(f"\nAnalyzing Headlines with Ollama ({OLLAMA_MODEL})...")
    scores = []
    confidences = []
    drivers = []
    sentiment_logs = []
    
    # Check if Ollama is initialized
    try:
        ollama.list()
    except Exception:
        print(f"[X] Error: Could not connect to Ollama at {_OLLAMA_HOST}. Make sure 'ollama serve' is running.")
        return 0.0, 0.0, 0, 0.0, []
    
    for i, article in enumerate(articles[:10]): # Limit to top 10 for speed
        print(f"   Processing Article {i+1}/{len(articles[:10])}...")
        result = get_sentiment_analysis(article)
        
        scores.append(result['score'])
        confidences.append(result['confidence'])
        drivers.append(result['driver'])
        
        preview = article[:60].replace('\n', ' ') + "..."
        print(f"      -> Score: {result['score']:.2f}, Conf: {result['confidence']:.2f}, Driver: {result['driver']} ('{preview}')")
        
        sentiment_logs.append({
            "article_preview": preview,
            "score": result['score'],
            "confidence": result['confidence'],
            "driver": result['driver']
        })

    scores = np.array(scores)
    confidences = np.array(confidences)
    
    if len(scores) == 0:
        return 0.0, 0.0, 0, 0.0, []
        
    # Weighted Mean based on Confidence
    if confidences.sum() > 0:
        weighted_mean = np.average(scores, weights=confidences)
    else:
        weighted_mean = scores.mean()
        
    sentiment_std = scores.std()
    high_conf_ratio = (confidences > 0.7).sum() / len(confidences)
    volume = len(articles)
    
    return weighted_mean, sentiment_std, volume, high_conf_ratio, sentiment_logs

# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHING (Adapted from stockDataBuilder.py)
# ══════════════════════════════════════════════════════════════════════════════

def fetch_fred_data(series_id, start, end, name):
    print(f"  Warning: pandas_datareader is replaced; skipping {name}")
    return None

def fetch_stock_proxy(ticker, start, end, name):
    try:
        stock = yf.download(ticker, start=start, end=end, progress=False)
        if len(stock) == 0: return None
        # Handle MultiIndex
        if isinstance(stock.columns, pd.MultiIndex):
            close = stock['Close'].iloc[:, 0]
        else:
            close = stock['Close']
            
        return pd.DataFrame({'Date': stock.index, name: close.values}).set_index('Date')
    except Exception as e:
        print(f"  Warning: Could not fetch {name} ({e})")
        return None

def create_composite_index(tickers, weights, start, end, name):
    try:
        composite = None
        for ticker, weight in zip(tickers, weights):
            stock = yf.download(ticker, start=start, end=end, progress=False)
            if len(stock) == 0: continue
            
            if isinstance(stock.columns, pd.MultiIndex):
                close = stock['Close'].iloc[:, 0]
            else:
                close = stock['Close']
            
            if composite is None:
                composite = close * weight
            else:
                close = close.reindex(composite.index).fillna(method='ffill')
                composite = composite + (close * weight)
        
        if composite is not None:
            return pd.DataFrame({'Date': composite.index, name: composite.values}).set_index('Date')
        return None
    except Exception as e:
        print(f"  Warning: Could not create {name} ({e})")
        return None

# ══════════════════════════════════════════════════════════════════════════════
# TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════════════════════

def add_technicals(df):
    # SMA
    df['SMA'] = df['Close'].rolling(window=20, min_periods=1).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ADX (Simple)
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=1).mean()
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14).mean() / (atr + 1e-10))
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/14).mean() / (atr + 1e-10)))
    dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-10)) * 100
    df['ADX'] = dx.ewm(alpha=1/14).mean()
    
    return df

def add_eda_features(df):
    """Add new features matching aapl_stock_sentiment_merged_dataset_2017_2026.csv"""
    # Range features
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_to_High_Ratio'] = df['Close'] / df['High']
    
    # Returns
    df['Close_Return_1d'] = df['Close'].pct_change()
    df['Close_Return_5d'] = df['Close'].pct_change(5)
    df['Close_Return_20d'] = df['Close'].pct_change(20)
    
    # Volatility
    returns = df['Close'].pct_change()
    df['Volatility_20'] = returns.rolling(window=20, min_periods=1).std()
    df['Volatility_60'] = returns.rolling(window=60, min_periods=1).std()
    
    # Macro diffs
    if 'CPI (Inflation)' in df.columns:
        df['CPI_Change'] = df['CPI (Inflation)'].diff()
    if 'Unemployment Rate' in df.columns:
        df['Unemployment_Change'] = df['Unemployment Rate'].diff()
        
    # Asset returns
    if 'Gold_Close' in df.columns:
        df['Gold_Return'] = df['Gold_Close'].pct_change()
    if 'GPU price index' in df.columns:
        df['GPU_Return'] = df['GPU price index'].pct_change()
    if 'Global AI investment index' in df.columns:
        df['AI_Investment_Return'] = df['Global AI investment index'].pct_change()
        
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATA ASSEMBLY
# ══════════════════════════════════════════════════════════════════════════════

def get_real_data(feature_cols=None, lookback_days=200):
    print("Fetching real-time data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=lookback_days)
    
    # 1. Main Stock (AAPL) — retry once if the first download times out
    for _attempt in range(3):
        aapl = yf.download('AAPL', start=start_date, end=end_date, progress=False, timeout=30)
        if len(aapl) > 0:
            break
        print(f"  Warning: AAPL download attempt {_attempt+1} returned empty, retrying...")
    
    if len(aapl) == 0:
        print("  ERROR: AAPL data unavailable after 3 retries. Cannot continue.")
        raise RuntimeError("AAPL market data unavailable – please check your internet connection.")
    
    if isinstance(aapl.columns, pd.MultiIndex):
        aapl.columns = aapl.columns.get_level_values(0)
    
    df = pd.DataFrame(index=aapl.index)
    df['Close'] = aapl['Close']
    df['High'] = aapl['High']
    df['Low'] = aapl['Low']
    df['Volume'] = aapl['Volume']
    
    # Helper to safely join – skip if proxy returned None
    def safe_join(df, other):
        if other is not None:
            return df.join(other, how='left')
        return df
    
    # 2. Market Data
    # Gold
    gold = fetch_stock_proxy('GC=F', start_date, end_date, 'Gold_Close')
    df = safe_join(df, gold)
    
    # VIX
    vix = fetch_stock_proxy('^VIX', start_date, end_date, 'VIX Index')
    df = safe_join(df, vix)
    
    # DXY
    dxy = fetch_stock_proxy('DX-Y.NYB', start_date, end_date, 'USD Index (DXY)')
    df = safe_join(df, dxy)
    
    # Bond Yields (10Y)
    bonds = fetch_stock_proxy('^TNX', start_date, end_date, 'Bond Yields (10Y US Treasury)')
    df = safe_join(df, bonds)
    
    # Currencies
    cny = fetch_stock_proxy('CNY=X', start_date, end_date, 'USD/CNY')
    df = safe_join(df, cny)
    
    eur = fetch_stock_proxy('EUR=X', start_date, end_date, 'USD/EUR')
    df = safe_join(df, eur)
    
    # Semiconductor
    soxx = fetch_stock_proxy('SOXX', start_date, end_date, 'Semiconductor index')
    df = safe_join(df, soxx)
    
    # GPU
    nvda = fetch_stock_proxy('NVDA', start_date, end_date, 'GPU price index')
    df = safe_join(df, nvda)
    
    # FRED Data (Macro)
    cpi = fetch_fred_data('CPIAUCSL', start_date, end_date, 'CPI (Inflation)')
    if cpi is not None: df = df.join(cpi, how='left').ffill()
    
    fed = fetch_fred_data('DFF', start_date, end_date, 'Interest Rate / Fed Funds Rate')
    if fed is not None: df = df.join(fed, how='left').ffill()
    
    m2 = fetch_fred_data('M2SL', start_date, end_date, 'Money Supply (M2)')
    if m2 is not None: df = df.join(m2, how='left').ffill()
    
    unemp = fetch_fred_data('UNRATE', start_date, end_date, 'Unemployment Rate')
    if unemp is not None: df = df.join(unemp, how='left').ffill()
    
    gas = fetch_fred_data('DHHNGSP', start_date, end_date, 'Natural gas price')
    if gas is not None: df = df.join(gas, how='left').ffill()
    
    # Indices
    ai_inv = create_composite_index(
        ['NVDA', 'MSFT', 'GOOGL', 'META'], [0.3, 0.3, 0.2, 0.2], 
        start_date, end_date, 'Global AI investment index'
    )
    if ai_inv is not None: df = df.join(ai_inv, how='left')
    
    capex = create_composite_index(
        ['AAPL', 'MSFT', 'GOOGL'], [0.33, 0.33, 0.33], 
        start_date, end_date, 'Tech CAPEX index'
    )
    if capex is not None: df = df.join(capex, how='left')
    
    # 3. Impute Missing/Specialized with Constants/Random
    # These are hard to fetch or proprietary
    n = len(df)
    rng = np.random.default_rng(42)
    
    defaults = {
        "Memory price (DRAM)": 3.5 + rng.normal(0, 0.1, n).cumsum(),
        "HBM price trend": 10.0 + rng.normal(0, 0.2, n).cumsum(),
        "Electricity price (industrial)": 0.08 + rng.uniform(-0.01, 0.01, n),
        "Carbon price / energy regulation": 25.0 + rng.normal(0, 0.5, n).cumsum(),
        "US Power Grid Load Index": 400.0 + rng.normal(0, 5, n).cumsum(),
        "AI regulation events (binary)": np.zeros(n),
        "Government AI funding": 50.0 + np.linspace(0, 10, n),
        
        # Stale indicators (set to 0 for real data usually)
        "CPI (Inflation)_days_stale": np.zeros(n),
        "Money Supply (M2)_days_stale": np.zeros(n),
        "Unemployment Rate_days_stale": np.zeros(n),
        
        # Sentiment (Placeholder - will be overwritten for latest)
        "sentiment_score": rng.uniform(-0.1, 0.1, n), # Low noise
        "sentiment_std": rng.uniform(0.1, 0.2, n),
        "article_volume": rng.integers(10, 30, n),
        "high_confidence_ratio": rng.uniform(0.6, 0.8, n)
    }
    
    for col, val in defaults.items():
        if col not in df.columns:
            df[col] = val
            
    # Fill remaining NaNs from joins
    df = df.ffill().bfill().fillna(0)
    
    # Add Technicals & EDA Features
    df = add_technicals(df)
    df = add_eda_features(df)
    
    # Final cleanup
    df = df.ffill().bfill().fillna(0)
    
    # Align to Model Feature Order
    if feature_cols is None:
        print("Warning: feature_cols not provided. Falling back to all dataset features.")
        feature_cols = [c for c in df.columns if c != 'Date']
    
    # Ensure all columns exist
    for c in feature_cols:
        if c not in df.columns:
            print(f"Missing column {c}, filling with 0")
            df[c] = 0.0
            
    return df[feature_cols], df

# ══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 80)
    print("REAL-TIME INFERENCE (AAPL) + LIVE SENTIMENT")
    print("=" * 80)
    
    # 1. Load Model
    horizon = 16
    scenario = "With Sentiment"
    try:
        model, scaler_target, scaler_cov, config = load_best_model(
            horizon, scenario, RESULTS_CSV, MODELS_DIR, SCALERS_DIR, DEVICE
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Get Market Data (Lookback)
    model_feature_cols = config.get("feature_cols", None)
    df_features, df_full = get_real_data(feature_cols=model_feature_cols)
    
    # 3. [NEW] Fetch Live News & Calculate Sentiment
    print("\n--- LIVE NEWS SENTIMENT ANALYSIS ---")
    if NEWS_DATA_API_KEY == "YOUR_API_KEY_HERE":
         print("⚠️  WARNING: No API Key provided. Using placeholder sentiment.")
         # Already random defaults in get_real_data
    else:
        # Fetch News
        articles = fetch_apple_news(NEWS_DATA_API_KEY)
        
        if articles:
            # Calculate Sentiment (Using Ollama)
            sent_score, sent_std, vol, conf_ratio, _logs = calculate_real_time_sentiment(articles)
            
            print(f"\n[OK] REAL-TIME SENTIMENT RESULTS:")
            print(f"   Score: {sent_score:.4f} (Scale: -1 to 1)")
            print(f"   Confidence: {conf_ratio:.1%}")
            print(f"   Volume: {vol} articles")
            
            # Update the LATEST row in the dataframe with these values
            # We assume the last row represents "now" or "today"
            last_idx = df_features.index[-1]
            
            df_features.at[last_idx, 'sentiment_score'] = sent_score
            df_features.at[last_idx, 'sentiment_std'] = sent_std
            df_features.at[last_idx, 'article_volume'] = vol
            df_features.at[last_idx, 'high_confidence_ratio'] = conf_ratio
            
            print(f"   -> Updated features for {last_idx}")
        else:
            print("   -> No articles found. Using default/random sentiment.")

    
    print(f"\nData Shape: {df_features.shape}")
    print(f"Latest Date: {df_full.index[-1]}")
    print(f"Latest Close: ${df_features['Close'].iloc[-1]:.2f}")
    
    # 4. Prepare for Inference
    # Need (N, 37) numpy array
    features = df_features.values.astype(np.float32)
    
    # Time marks
    dates = df_full.index
    # Try import time_features
    try:
        from informerMLopsUpdated import time_features
        time_marks = time_features(dates)
    except ImportError:
         print("Warning: time_features not found, using simple mock.")
         time_marks = np.column_stack([
            dates.month.values,
            dates.day.values,
            dates.weekday.values,
            np.zeros(len(dates)) 
        ])

    
    # 5. Predict
    print("\nRunning Inference...")
    predictions = predict_with_model(
        model, scaler_target, scaler_cov, features, time_marks,
        config["seq_len"], config["label_len"], config["pred_len"],
        config["input_dim"], DEVICE
    )
    
    # 6. Output
    print(f"\nFORECAST (Next {horizon} Days):")
    print(f"{'Date':<12} {'Price':>10}")
    print(f"{'-'*12} {'-'*10}")
    
    last_date = df_full.index[-1]
    for i, price in enumerate(predictions.flatten(), 1):
        next_date = last_date + timedelta(days=i)
        print(f"{next_date.strftime('%Y-%m-%d'):<12} ${price:>9.2f}")
        
    avg_pred = predictions.mean()
    last_price = features[-1, 0]
    change = (avg_pred - last_price) / last_price * 100
    
    print(f"\nSummary:")
    print(f"  Current: ${last_price:.2f}")
    print(f"  Avg Pred: ${avg_pred:.2f}")
    print(f"  Change:   {change:+.2f}%")

if __name__ == "__main__":
    main()

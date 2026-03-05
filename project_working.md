# Data Pipeline Architecture

The data pipeline consists of several interconnected stages, bringing together external datasets and processing them iteratively. Based on the DAG structures represented in DagsHub, the data pipeline flow is strictly defined as follows:

![Data Pipeline Top Half](./docs/dag_1.png)
![Data Pipeline Bottom Half](./docs/dag_2.png)

## Pipeline Stages

### 1. extract_market_data
* **Inputs:** `extract_market_data.py` (Git Managed File)
* **Outputs:** `apple_technical_macro_features.csv` (DVC Managed File)
* **Core Logic:** Calculates 4 key technical indicators (SMA, RSI, OBV, ADX) and merges them with Gold and USD/JPY macroeconomic features.
```python
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    # 1. SMA
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    # 2. RSI
    # ... calculates average gain / loss over 14 days
    # 3. OBV
    df_stock['OBV'] = (np.sign(df_stock['Close'].diff()) * df_stock['Volume']).fillna(0).cumsum()
    # 4. Corrected ADX
    # ... calculates positive / negative directional movement (plus_di, minus_di)
```

### 2. merge_news_1
* **Inputs:**
  * `merge_news_1.py` (Git Managed File)
  * `apple_technical_macro_features.csv` (DVC Managed File)
  * `apple_news_data.csv` (External File)
* **Outputs:** `kaggle_technical_news.csv` (DVC Managed File)
* **Core Logic:** Aligns external timestamped news to the correct trading days on the stock market, carrying over articles posted after 16:00 ET to the next valid open day.
```python
def get_applicable_trading_day(ts):
    if pd.isnull(ts): return None
    news_date = ts.date()
    # Handling 16:00 ET after-hours cutoff
    if ts.hour >= 16:
        news_date += timedelta(days=1)
    for trading_day in valid_trading_days:
        if trading_day >= news_date:
            return trading_day
    return None
```

### 3. merge_news_2
* **Inputs:**
  * `merge_news_2.py` (Git Managed File)
  * `kaggle_technical_news.csv` (DVC Managed File)
  * `news.csv` (External File)
* **Outputs:** `kaggle_github_technical_news.csv` (DVC Managed File)

### 4. merge_news_3
* **Inputs:**
  * `merge_news_3.py` (Git Managed File)
  * `kaggle_github_technical_news.csv` (DVC Managed File)
  * `all_the_news_2.000.csv` to `...2.003.csv` (External Files)
* **Outputs:**
  * `processing_checklist.json` (DVC Managed File)
  * `all_news_merged_uncleaned.csv` (DVC Managed File)
* **Core Logic:** Performs chunked processing to safely load massive, gigabyte-scale datasets. Utilizes `json` checklists to ensure fault-tolerance and resume capability.
```python
# Processing massive files in chunks to avoid blowing up system memory
for chunk in pd.read_csv(
    news_file, 
    chunksize=100000, 
    engine='python',             
    encoding_errors='replace',   
    on_bad_lines='skip',         
    quoting=3                    
):
```

### 5. data_ingestion
* **Inputs:**
  * `data_ingestion.py` (Git Managed File)
  * `all_news_merged_uncleaned.csv` (DVC Managed File)
* **Outputs:** `final_raw_dataset.csv` (DVC Managed File)
* **Core Logic:** Emits a final validation report confirming the shape and date range of the aggregated data before officially committing it to the raw dataset folder.

### 6. data_preprocessing
* **Inputs:**
  * `data_preprocessing.py` (Git Managed File)
  * `final_raw_dataset.csv` (DVC Managed File)
* **Outputs:**
  * `cleaning_state.json` (DVC Managed File)
  * `cleaned_news_features.csv` (DVC Managed File)
* **Core Logic:** Enforces a three-tier extreme filtering logic: checking word length constraints, strictly deduplicating text, and requiring headlines to mention either Apple explicitly or both a macro-economic concept AND a tech supply chain concept.
```python
# EXTREME FILTER 3: The Two-Tier Logic Gate
has_apple = bool(PATTERN_APPLE.search(h_lower))

if has_apple:
    cleaned_headlines.append(cleaned_h)
    continue
    
has_tech = bool(PATTERN_TECH.search(h_lower))
has_macro = bool(PATTERN_MACRO.search(h_lower))

if has_tech and has_macro:
    cleaned_headlines.append(cleaned_h)
```

### 7. feature_engineering
* **Inputs:**
  * `feature_engineering.py` (Git Managed File)
  * `cleaned_news_features.csv` (DVC Managed File)
* **Outputs:** `final_sentiment_scored_dataset.csv` (DVC Managed File)
* **Core Logic:** Iterates over the cleaned headlines, routing them to a local `Llama 3` model in GGUF format via `llama_cpp`. It uses a cached, structured prompt to forcefully extract 'positive', 'negative', or 'neutral' sentiments and maps them to numerical averages.
```python
prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a financial analyst. Classify the sentiment of this news headline.
Return ONLY one word: 'positive', 'negative', or 'neutral'.<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Headline: "{safe_headline}"<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Sentiment:"""

# Llama 3 Inference on Local CPU/GPU
output = llm(
    prompt,
    max_tokens=5,
    temperature=0.0, 
    stop=["<|eot_id|>"], 
    echo=False
)
```

## DVC (Data Version Control) Beginner's Guide

This project relies entirely on **DVC** to manage its data and machine learning pipeline. If you're new to DVC, think of it as "Git for Data." While Git tracks code changes, DVC tracks large datasets and the steps required to process them. 

### Why do we use DVC here?
* **Reproducibility**: DVC knows exactly which scripts (`.py`), parameters (`params.yaml`), and data (`.csv`) correspond to each other.
* **Smart Execution**: If you change the preprocessing script, DVC will only rerun preprocessing and feature engineering. It will *skip* extraction and merging because those haven't changed!

### Essential DVC Commands

Here are the most common commands you'll need when interacting with this project:

#### 1. Run the Pipeline
To execute the pipeline and generate the final dataset, simply run:
```bash
dvc repro
```
*If everything is up-to-date, DVC will say `Data and pipelines are up to date` and exit.*

#### 2. Run a Specific Stage
If you only want to test a specific part of the pipeline (e.g., you changed the technical indicator logic and want to test data extraction):
```bash
dvc repro extract_market_data
```

#### 3. Check Pipeline Status
To see which stages have changed or need to be rerun:
```bash
dvc status
```

#### 4. Managing Data (Remote Storage)
If the project is connected to remote storage (like an AWS S3 bucket, Google Drive, or DagsHub storage), you can pull the latest data without having to run the scripts yourself:
```bash
# Pull datasets from the remote server
dvc pull

# Push your newly generated datasets to the remote server
dvc push
```

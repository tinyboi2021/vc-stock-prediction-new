# Data Pipeline Architecture

The data pipeline consists of several interconnected stages, bringing together external datasets and processing them iteratively. Based on the DAG structures represented in DagsHub, the data pipeline flow is strictly defined as follows:

![Data Pipeline Top Half](./docs/dag_1.png)
![Data Pipeline Bottom Half](./docs/dag_2.png)

## Pipeline Stages

### 1. scrape_news
* **Inputs:**
  * `src/data/scrape_news.py` (Git Managed File)
* **Outputs:** `data/raw/apple_news_data.csv` (DVC Managed File)
* **Core Logic:** Aggregates news articles computationally from multiple APIs (NYT, Guardian, GNews, Finnhub, Alpha Vantage) utilizing a layered "skeleton and flesh" approach to construct a deeply contextual historical dataset.

### 2. calculate_sentiment
* **Inputs:**
  * `src/features/calculate_sentiment.py` (Git Managed File)
  * `data/raw/apple_news_data.csv` (DVC Managed File)
* **Outputs:** `data/processed/apple_news_sentiment.csv` (DVC Managed File)
* **Core Logic:** Passes the scraped headlines/articles into a local Llama 3 model in GGUF format via `llama_cpp`. It uses a strictly formatted prompt to analyze both general confidence-calibrated sentiment and Apple-specific economic sentiment, creating a weighted daily sentiment score.

### 3. build_datasets
* **Inputs:**
  * `src/data/build_datasets.py` (Git Managed File)
  * `data/processed/apple_news_sentiment.csv` (DVC Managed File)
* **Outputs:** 
  * `data/processed/dataset_small.csv` (DVC Managed File)
  * `data/processed/dataset_large.csv` (DVC Managed File)
* **Core Logic:** Generates robust technical indicators (SMA, RSI, OBV, ADX), fetches multi-asset references (Gold, USD/JPY), macroeconomic contexts (CPI, Fed Rates, Unemployment), and synthesizes them with the LLM-derived sentiment. Crucially, it emits exactly two required datasets:
  * **Version 1 (Small):** `Date, Close, SMA, RSI, OBV, ADX, Gold_Close, USD_JPY_Close, sentiment_score`
  * **Version 2 (Large):** 28 highly specialized features including stationarity transformations (Returns), AI regulation dummy events, Market volatility scores, Tech sector trends, and Supply chain disruptions.

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

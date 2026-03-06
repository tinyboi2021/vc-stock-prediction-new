# Data Pipeline Redesign Walkthrough

The Apple stock prediction data pipeline has been successfully streamlined and modernized according to the new requirements. The legacy 7-stage architecture was completely replaced with a modular, 3-stage flow.

## 1. Scripts Creation and Adaptation

We systematically adapted the provided reference scripts to fit the current project constraints:
1. **Scraping (`src/data/scrape_news.py`)**: Extracted from `multiApiNewsArticleScraping.py`. Handles multi-API parallel news scraping specifically optimized for Apple (AAPL).
2. **Sentiment Analysis (`src/features/calculate_sentiment.py`)**: Extracted from `sentimentScoreNewsArticle.py`. Ingests the news text and passes it to the local Llama 3 GGUF model via `llama_cpp`, outputting aggregated daily sentiment scores.
3. **Dataset Synthesis (`src/data/build_datasets.py`)**: Adapted from `stockDataBuilder.py`. Downloads stock data, technical indicators, commodities (Gold, USD/JPY), and custom FRED macroeconomic indicators (CPI, Interest Rates, VIX). It merges everything with the sentiment scores.

## 2. DVC Pipeline Integration

The `dvc.yaml` was completely refactored. We stripped out the obsolete node commands (`extract_market_data`, `merge_news_1`, `data_ingestion`, etc.) and replaced them with:

```yaml
stages:
  scrape_news:
    cmd: python src/data/scrape_news.py
    deps:
      - src/data/scrape_news.py
    outs:
      - data/raw/apple_news_data.csv:
          cache: true
  
  calculate_sentiment:
    cmd: python src/features/calculate_sentiment.py
    deps:
      - src/features/calculate_sentiment.py
      - data/raw/apple_news_data.csv
    outs:
      - data/processed/apple_news_sentiment.csv:
          cache: true

  build_datasets:
    cmd: python src/data/build_datasets.py
    deps:
      - src/data/build_datasets.py
      - data/processed/apple_news_sentiment.csv
    outs:
      - data/processed/dataset_small.csv:
          cache: true
      - data/processed/dataset_large.csv:
          cache: true
```

## 3. Dataset Generation and Verification

The `build_datasets.py` pipeline was executed locally to confirm its operational status. It successfully emitted the requested custom datasets into the `data/processed/` directory.

- **`dataset_small.csv` Outputs**: `Date, Close, SMA, RSI, OBV, ADX, Gold_Close, USD_JPY_Close, sentiment_score`
- **`dataset_large.csv` Outputs**: 28 total attributes incorporating the complex, customized stationarity modifiers, VIX indices, and categorical event markers (like *Federal reserve announcements* and *Supply chain disruption scores*).

Finally, `project_working.md` was rewritten to document the new 3-stage model.

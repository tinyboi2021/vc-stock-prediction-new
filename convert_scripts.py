import os

# 1. Adapt multiApiNewsArticleScraping.py
with open('reference codes/multiApiNewsArticleScraping.py', 'r', encoding='utf-8') as f:
    scrape_code = f.read()

scrape_code = scrape_code.replace('COMPANY = "Nvidia"', 'COMPANY = "Apple"')
scrape_code = scrape_code.replace('TICKER = "NVDA"', 'TICKER = "AAPL"')
scrape_code = scrape_code.replace('Nvidia OR NVDA OR GeForce OR RTX OR \\"Jensen Huang\\"', 'Apple OR AAPL OR iPhone OR MacBook OR \\"Tim Cook\\"')
scrape_code = scrape_code.replace('Nvidia OR NVDA OR GeForce OR RTX', 'Apple OR AAPL OR iPhone OR Mac')
scrape_code = scrape_code.replace('Nvidia OR NVDA OR GeForce OR \\"Jensen Huang\\"', 'Apple OR AAPL OR iPhone OR Mac OR \\"Tim Cook\\"')
scrape_code = scrape_code.replace('INPUT_DATASET = "/kaggle/input/nvidia-articles-scrape-bulk/NVDA_2017_2025_LOGGED_BULK.csv"', 'INPUT_DATASET = "data/raw/AAPL_2017_2026_LOGGED_BULK.csv"')
scrape_code = scrape_code.replace('OUTPUT_FILE  = "/kaggle/working/NVDA_2017_2025_LOGGED_BULK.csv"', 'OUTPUT_FILE = "data/raw/apple_news_data.csv"')

with open('src/data/scrape_news.py', 'w', encoding='utf-8') as f:
    f.write(scrape_code)
print('Created src/data/scrape_news.py')

# 2. Adapt sentimentScoreNewsArticle.py
with open('reference codes/sentimentScoreNewsArticle.py', 'r', encoding='utf-8') as f:
    sent_code = f.read()

sent_code = sent_code.replace('input_file_path = "/kaggle/input/apple-final-dataset-merged/merged_financial_news_cleaned_final_merge.csv"', 'input_file_path = "data/raw/apple_news_data.csv"')
sent_code = sent_code.replace('previous_output_path = "/kaggle/input/appl-news-article-dataset/sentiment_scored_article_dataset.csv"', 'previous_output_path = "data/processed/apple_news_sentiment.csv"')
sent_code = sent_code.replace('output_file_path = "/kaggle/working/sentiment_scored_article_dataset.csv"', 'output_file_path = "data/processed/apple_news_sentiment.csv"')

with open('src/features/calculate_sentiment.py', 'w', encoding='utf-8') as f:
    f.write(sent_code)
print('Created src/features/calculate_sentiment.py')

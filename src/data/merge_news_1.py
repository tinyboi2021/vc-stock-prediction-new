import pandas as pd
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_and_prep_stock_dates(stock_file_path: str) -> tuple[pd.DataFrame, str, list]:
    """Loads stock data and extracts valid trading days."""
    logging.info(f"Loading stock dataset from {stock_file_path}...")
    df_stock = pd.read_csv(stock_file_path)
    
    stock_date_col = df_stock.columns[0] 
    df_stock['date_only'] = pd.to_datetime(df_stock[stock_date_col]).dt.date
    
    valid_trading_days = sorted(df_stock['date_only'].dropna().unique())
    return df_stock, stock_date_col, valid_trading_days

def load_and_prep_news_dates(news_file_path: str) -> tuple[pd.DataFrame, str]:
    """Loads news data, isolates required columns, and standardizes to US/Eastern time."""
    logging.info(f"Loading news dataset from {news_file_path}...")
    
    df_news = pd.read_csv(
        news_file_path, 
        encoding='latin-1', 
        on_bad_lines='skip'
    )
    
    news_date_col, news_text_col = df_news.columns[0], df_news.columns[1]
    df_news = df_news[[news_date_col, news_text_col]].copy()
    
    logging.info("Standardizing news timestamps to US/Eastern...")
    
    # THE FIX: Add errors='coerce' to gracefully handle shifted text rows
    df_news['datetime_utc'] = pd.to_datetime(
        df_news[news_date_col], 
        format='mixed', 
        utc=True, 
        errors='coerce' # <--- This prevents the crash
    )
    
    # Now, simply drop the rows that failed to parse into valid dates
    df_news = df_news.dropna(subset=['datetime_utc'])
    
    df_news['datetime_et'] = df_news['datetime_utc'].dt.tz_convert('US/Eastern')
    
    return df_news, news_text_col
    
    news_date_col, news_text_col = df_news.columns[0], df_news.columns[1]
    df_news = df_news[[news_date_col, news_text_col]].copy()
    
    logging.info("Standardizing news timestamps to US/Eastern...")
    df_news['datetime_utc'] = pd.to_datetime(df_news[news_date_col], format='mixed', utc=True)
    df_news['datetime_et'] = df_news['datetime_utc'].dt.tz_convert('US/Eastern')
    
    return df_news, news_text_col

def map_news_to_trading_days(df_news: pd.DataFrame, valid_trading_days: list) -> pd.DataFrame:
    """Maps news timestamps to the correct valid trading day, handling after-hours."""
    logging.info("Mapping news to valid trading days (handling 16:00 ET after-hours cutoff)...")
    
    def get_applicable_trading_day(ts):
        if pd.isnull(ts):
            return None
        news_date = ts.date()
        if ts.hour >= 16:
            news_date += timedelta(days=1)
        for trading_day in valid_trading_days:
            if trading_day >= news_date:
                return trading_day
        return None

    df_news['target_trading_date'] = df_news['datetime_et'].apply(get_applicable_trading_day)
    return df_news.dropna(subset=['target_trading_date'])

def aggregate_and_merge(df_stock: pd.DataFrame, df_news: pd.DataFrame, 
                        stock_date_col: str, news_text_col: str) -> pd.DataFrame:
    """Aggregates daily headlines and left-joins them onto the stock dataset."""
    logging.info("Aggregating headlines by trading day...")
    
    aggregated_news = df_news.groupby('target_trading_date')[news_text_col].apply(
        lambda headlines: ' | '.join(headlines.dropna().astype(str).str.strip())
    ).reset_index()
    
    aggregated_news.rename(columns={
        'target_trading_date': 'date_only', 
        news_text_col: 'news_headlines'
    }, inplace=True)

    logging.info("Merging news into the stock dataset...")
    final_df = pd.merge(df_stock, aggregated_news, on='date_only', how='left')
    
    # Clean up
    final_df = final_df.drop_duplicates(subset=[stock_date_col])
    final_df = final_df.drop(columns=['date_only'])
    
    return final_df

def process_kaggle_merge(news_file_path: str, stock_file_path: str, output_file_path: str):
    """Orchestrator function to run the Kaggle news merging pipeline."""
    df_stock, stock_date_col, valid_trading_days = load_and_prep_stock_dates(stock_file_path)
    df_news, news_text_col = load_and_prep_news_dates(news_file_path)
    
    df_mapped_news = map_news_to_trading_days(df_news, valid_trading_days)
    
    final_df = aggregate_and_merge(df_stock, df_mapped_news, stock_date_col, news_text_col)
    
    # Save the output
    final_df.to_csv(output_file_path, index=False)
    logging.info(f"Success! Interim dataset saved to {output_file_path}")
    print(final_df.head())

if __name__ == "__main__":
    import yaml
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["merge_news_1"]
    
    process_kaggle_merge(
        params["news_dataset_path"], 
        params["stock_dataset_path"], 
        params["output_path"]
    )
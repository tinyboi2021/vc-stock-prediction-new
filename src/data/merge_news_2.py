import pandas as pd
import numpy as np
from datetime import timedelta
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def map_news_to_trading_days(df_main: pd.DataFrame, df_news: pd.DataFrame) -> pd.DataFrame:
    """Standardizes timezones and maps news to the correct trading day."""
    logging.info("Mapping new news timestamps to valid trading days...")
    
    # Setup target dates from main dataset
    date_col_main = df_main.columns[0]
    df_main['target_date'] = pd.to_datetime(df_main[date_col_main]).dt.date
    valid_trading_days = sorted(df_main['target_date'].dropna().unique())

    # Standardize news dates
    date_col_new = df_news.columns[0]
    
    # Pre-emptively handling bad date formats from scraped text using errors='coerce'
    parsed_dates = pd.to_datetime(df_news[date_col_new].astype(str), format='mixed', errors='coerce')
    
    # Assign the parsed dates back to a new column
    df_news['datetime_utc'] = parsed_dates
    
    # Drop any rows where the date was actually just broken text
    df_news = df_news.dropna(subset=['datetime_utc']).copy()
    
    if df_news['datetime_utc'].dt.tz is None:
        df_news['datetime_utc'] = df_news['datetime_utc'].dt.tz_localize('UTC')
        
    df_news['datetime_et'] = df_news['datetime_utc'].dt.tz_convert('US/Eastern')

    def get_applicable_trading_day(ts):
        if pd.isnull(ts):
            return None
        news_date = ts.date()
        if ts.hour >= 16:  # After market close
            news_date += timedelta(days=1)
        for trading_day in valid_trading_days:
            if trading_day >= news_date:
                return trading_day
        return None

    df_news['target_date'] = df_news['datetime_et'].apply(get_applicable_trading_day)
    return df_news.dropna(subset=['target_date'])

def combine_news_safely(row: pd.Series) -> str:
    """Combines old and new headlines, preventing duplicates."""
    existing_str = str(row['news_headlines']).strip() if pd.notnull(row.get('news_headlines', np.nan)) else ""
    new_str = str(row['new_appended_news']).strip() if pd.notnull(row.get('new_appended_news', np.nan)) else ""
    
    existing_str = "" if existing_str.lower() == "nan" else existing_str
    new_str = "" if new_str.lower() == "nan" else new_str

    if not existing_str and not new_str:
        return np.nan
        
    existing_headlines = [h.strip() for h in existing_str.split('|') if h.strip()]
    new_headlines = [h.strip() for h in new_str.split('|') if h.strip()]
    
    combined_headlines = existing_headlines.copy()
    for headline in new_headlines:
        if headline not in combined_headlines:
            combined_headlines.append(headline)
            
    return " | ".join(combined_headlines)

def process_github_merge(main_path: str, news_path: str, output_path: str):
    """Orchestrates the loading, mapping, merging, and saving of the datasets."""
    logging.info(f"Loading main interim dataset: {main_path}")
    df_main = pd.read_csv(main_path)
    
    logging.info(f"Loading new GitHub news dataset: {news_path}")
    # Using latin-1 and skipping bad lines for scraped data safety
    df_new_news = pd.read_csv(news_path, encoding='latin-1', on_bad_lines='skip')
    
    news_col_new = df_new_news.columns[1]

    # 1. Map dates
    df_new_news = map_news_to_trading_days(df_main, df_new_news)

    # 2. Aggregate new news
    logging.info("Aggregating new headlines by trading day...")
    agg_new_news = df_new_news.groupby('target_date')[news_col_new].apply(
        lambda headlines: ' | '.join(headlines.dropna().astype(str).str.strip())
    ).reset_index()
    agg_new_news.rename(columns={news_col_new: 'new_appended_news'}, inplace=True)

    # 3. Merge into main dataset
    logging.info("Merging datasets...")
    df_main = pd.merge(df_main, agg_new_news, on='target_date', how='left')
    if 'news_headlines' not in df_main.columns:
        df_main['news_headlines'] = np.nan

    # 4. Apply deduplication
    logging.info("Applying safe combinations to prevent duplicate headlines...")
    df_main['news_headlines'] = df_main.apply(combine_news_safely, axis=1)

    # 5. Clean up and Save
    df_main = df_main.drop(columns=['target_date', 'new_appended_news'], errors='ignore')
    
    # Save to the NEW output path instead of overwriting
    df_main.to_csv(output_path, index=False)
    logging.info(f"Success! Deduplicated dataset saved to {output_path}")

if __name__ == "__main__":
    import yaml
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["merge_news_2"]
        
    process_github_merge(
        params["main_dataset_path"], 
        params["news_dataset_path"], 
        params["output_path"]
    )
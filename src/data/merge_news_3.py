import pandas as pd
import numpy as np
from datetime import timedelta
import os
import json
import logging
import shutil
import csv
# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def combine_news_safely(existing_str, new_str) -> str:
    """Safe combination logic to prevent duplicates."""
    if pd.isna(existing_str) or not str(existing_str).strip() or str(existing_str).lower() == "nan":
        return new_str if pd.notna(new_str) and str(new_str).lower() != "nan" else np.nan
    if pd.isna(new_str) or not str(new_str).strip() or str(new_str).lower() == "nan":
        return existing_str

    existing_list = [h.strip() for h in str(existing_str).split('|') if h.strip()]
    existing_set = set(existing_list)
    new_list = [h.strip() for h in str(new_str).split('|') if h.strip()]

    for h in new_list:
        if h not in existing_set:
            existing_list.append(h)
            existing_set.add(h) 
            
    return " | ".join(existing_list)

def get_applicable_trading_day(ts, valid_trading_days):
    """Maps to the next valid trading day, handling 16:00 ET cutoff."""
    if pd.isnull(ts): return None
    news_date = ts.date()
    if ts.hour >= 16:
        news_date += timedelta(days=1)
    for trading_day in valid_trading_days:
        if trading_day >= news_date:
            return trading_day
    return None

def process_large_news_files(input_main_path: str, output_main_path: str, news_file_paths: list, checkpoint_file: str):
    """Orchestrates chunked merging of massive datasets with checkpointing."""
    
    # 1. Prepare the working file
    # We copy the input to the output path so we don't overwrite our source of truth
    if not os.path.exists(output_main_path):
        logging.info(f"Creating working copy at {output_main_path}...")
        shutil.copy2(input_main_path, output_main_path)

    # 2. Initialize or Load Checklist
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checklist = json.load(f)
        logging.info("Loaded existing checklist. Resuming progress...")
    else:
        checklist = {file: False for file in news_file_paths}

    # 3. Load the working dataset
    df_main = pd.read_csv(output_main_path)
    date_col_main = df_main.columns[0]
    df_main['target_date'] = pd.to_datetime(df_main[date_col_main]).dt.date
    valid_trading_days = sorted(df_main['target_date'].dropna().unique())
    
    if 'news_headlines' not in df_main.columns:
        df_main['news_headlines'] = np.nan

    # 4. Process files in chunks
    for news_file in news_file_paths:
        if checklist.get(news_file, False):
            logging.info(f"⏭️ Skipping {news_file} (already complete).")
            continue
            
        logging.info(f"⏳ Processing {news_file} in chunks...")
        
        chunk_size = 100000 
        processed_chunks = []
        
        # Added NLP safeguards: encoding and on_bad_lines
        for chunk in pd.read_csv(
            news_file, 
            chunksize=chunk_size, 
            engine='python',             # Keep the flexible Python engine
            encoding_errors='replace',   # Prevent weird character crashes
            on_bad_lines='skip',         # Drop rows that get messed up
            quoting=3                    # THE FIX: 3 means QUOTE_NONE. Quotes are now just regular text.
        ):
            date_col_new, news_col_new = chunk.columns[0], chunk.columns[1]
            chunk = chunk[[date_col_new, news_col_new]].copy()
            
            # Standardize dates
            parsed_dates = pd.to_datetime(chunk[date_col_new].astype(str), format='mixed', errors='coerce', utc=True)
            chunk['datetime_et'] = parsed_dates.dt.tz_convert('US/Eastern')
            chunk['target_date'] = chunk['datetime_et'].apply(lambda ts: get_applicable_trading_day(ts, valid_trading_days))
            chunk = chunk.dropna(subset=['target_date'])
            
            # Deduplicate within chunk
            chunk = chunk.drop_duplicates(subset=['target_date', news_col_new])
            processed_chunks.append(chunk[['target_date', news_col_new]])
            
        if processed_chunks:
            df_file_agg = pd.concat(processed_chunks, ignore_index=True)
            df_file_agg = df_file_agg.drop_duplicates(subset=['target_date', news_col_new])
            
            # Now we group and join
            df_file_agg = df_file_agg.groupby('target_date')[news_col_new].apply(
                # THE FIX: Added .dropna() and .str.strip() to handle blank/float headlines
                lambda headlines: ' | '.join(headlines.dropna().astype(str).str.strip())
            ).reset_index()
            
            df_file_agg.rename(columns={news_col_new: 'new_appended_news'}, inplace=True)
            
            # Merge into main dataset
            df_main = pd.merge(df_main, df_file_agg, on='target_date', how='left')
            
            df_main['news_headlines'] = [
                combine_news_safely(ex, nw)
                for ex, nw in zip(df_main['news_headlines'], df_main['new_appended_news'])
            ]
            
            df_main = df_main.drop(columns=['new_appended_news'])
            
        # 5. Overwrite the WORKING file, not the input file
        df_main_out = df_main.drop(columns=['target_date'])
        df_main_out.to_csv(output_main_path, index=False)
        
        # 6. Update Checklist
        checklist[news_file] = True
        with open(checkpoint_file, 'w') as f:
            json.dump(checklist, f)
            
        logging.info(f"✅ Successfully processed {news_file} and updated checkpoint.")
        
    logging.info("🎉 All massive news files have been successfully aggregated and merged!")

if __name__ == "__main__":
    import yaml
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["merge_news_3"]

    process_large_news_files(
        params["input_main_path"], 
        params["output_main_path"], 
        params["large_news_files"], 
        params["checkpoint_file"]
    )
import pandas as pd
import numpy as np
import re
import os
import json
import logging
import yaml

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- CONFIGURATION & REGEX COMPILATION ---
APPLE_DIRECT = {
    'apple', 'aapl', 'iphone', 'ipad', 'macbook', 'imac', 'mac mini', 'mac pro', 
    'airpods', 'apple watch', 'homepod', 'airtag', 'vision pro', 'wearables',
    'ios', 'ipados', 'macos', 'watchos', 'tvos', 'siri', 'app store', 'icloud', 
    'apple pay', 'apple music', 'apple tv+', 'apple card', 'apple intelligence',
    'tim cook', 'jony ive', 'luca maestri', 'craig federighi'
}

TECH_SUPPLY = {
    'foxconn', 'tsmc', 'taiwan semiconductor', 'qualcomm', 'broadcom', 'samsung', 
    'nvidia', 'intel', 'semiconductor', 'chip', 'arm', 'asml', 'micron', 'pegatron', 
    'luxshare', 'wistron', 'tata', 'goertek', 'huawei', 'microsoft', 'google', 
    'alphabet', 'meta', 'tesla', 'smartphone', 'tech', 'nasdaq'
}

MACRO_FINANCE = {
    'earnings', 'revenue', 'guidance', 'profit', 'stock', 'shares', 'wall street', 
    'dividend', 'buyback', 'share repurchase', 'stock split', 'market cap', 'trillion', 
    'downgrade', 'upgrade', 'price target', 'fed', 'federal reserve', 'interest rate', 
    'powell', 'jerome powell', 'inflation', 'cpi', 'rate cut', 'rate hike', 'central bank', 
    'gdp', 'fomc', 'recession', 'tariff', 'trade war', 'geopolitical', 'sanctions', 
    'antitrust', 'doj', 'eu', 'export ban', 'monopoly', 'patent', 'shortage'
}

PATTERN_APPLE = re.compile(r'\b(?:' + '|'.join(map(re.escape, APPLE_DIRECT)) + r')\b', re.IGNORECASE)
PATTERN_TECH = re.compile(r'\b(?:' + '|'.join(map(re.escape, TECH_SUPPLY)) + r')\b', re.IGNORECASE)
PATTERN_MACRO = re.compile(r'\b(?:' + '|'.join(map(re.escape, MACRO_FINANCE)) + r')\b', re.IGNORECASE)

# --- CORE LOGIC ---
def extreme_clean_and_filter(text) -> str:
    """Filters headlines based on length, duplicates, and financial relevance."""
    if pd.isna(text) or not str(text).strip() or str(text).lower() == "nan":
        return np.nan
        
    headlines = str(text).split('|')
    cleaned_headlines = []
    seen = set()
    
    for h in headlines:
        # Light Scrub
        cleaned_h = re.sub(r'[^a-zA-Z0-9\s\-\.\$\%]', '', h).strip()
        cleaned_h = re.sub(r'\s+', ' ', cleaned_h)
        
        # EXTREME FILTER 1: Length Constraint
        word_count = len(cleaned_h.split())
        if word_count < 4 or word_count > 50:
            continue
            
        h_lower = cleaned_h.lower()
        
        # EXTREME FILTER 2: Strict Deduplication
        if h_lower in seen:
            continue
        seen.add(h_lower)
        
        # EXTREME FILTER 3: The Two-Tier Logic Gate
        has_apple = bool(PATTERN_APPLE.search(h_lower))
        
        if has_apple:
            cleaned_headlines.append(cleaned_h)
            continue
            
        has_tech = bool(PATTERN_TECH.search(h_lower))
        has_macro = bool(PATTERN_MACRO.search(h_lower))
        
        if has_tech and has_macro:
            cleaned_headlines.append(cleaned_h)
            
    if not cleaned_headlines:
        return np.nan
        
    return ' | '.join(cleaned_headlines)


def process_headlines_safely(input_file: str, output_file: str, checkpoint_file: str, chunk_size: int):
    """Processes large datasets in chunks with save-state checkpointing."""
    start_chunk = 0
    
    # THE BUG FIX: Actually load the checkpoint if it exists!
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, 'r') as f:
                state = json.load(f)
                start_chunk = state.get('last_processed_chunk', 0)
                logging.info(f"🔄 Resuming from chunk {start_chunk}...")
        except json.JSONDecodeError:
            logging.warning("Checkpoint file corrupted. Starting from scratch.")

    chunk_iterator = pd.read_csv(input_file, chunksize=chunk_size, low_memory=False)
    
    for i, chunk in enumerate(chunk_iterator):
        if i < start_chunk:
            continue
            
        logging.info(f"⏳ Scrubbing and filtering chunk {i + 1} with EXTREME limits...")
        
        if 'news_headlines' in chunk.columns:
            chunk['news_headlines'] = chunk['news_headlines'].apply(extreme_clean_and_filter)
            
        # Write to file
        mode = 'w' if i == 0 and start_chunk == 0 else 'a'
        header = True if i == 0 and start_chunk == 0 else False
        chunk.to_csv(output_file, mode=mode, header=header, index=False)
        
        # Update checkpoint
        with open(checkpoint_file, 'w') as f:
            json.dump({'last_processed_chunk': i + 1}, f)
            
    logging.info(f"🎉 EXTREME Preprocessing complete! Cleaned dataset saved to: {output_file}")


if __name__ == "__main__":
    try:
        with open("params.yaml", "r") as f:
            params = yaml.safe_load(f)["preprocess_data"]
            
        process_headlines_safely(
            params["input_path"], 
            params["output_path"], 
            params["checkpoint_file"], 
            params["chunk_size"]
        )
        
    except FileNotFoundError:
        logging.error("params.yaml not found.")
        exit(1)
    except KeyError as e:
        logging.error(f"Missing key in params.yaml: {e}")
        exit(1)
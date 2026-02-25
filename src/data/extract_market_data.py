import yfinance as yf
import pandas as pd
import numpy as np
import logging

# Set up logging so we can track the execution in the terminal
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def fetch_stock_data(ticker: str, fetch_start_date: str, end_date: str) -> pd.DataFrame:
    """Fetches historical stock data using the runway date."""
    logging.info(f"Fetching stock data for {ticker} from {fetch_start_date} to {end_date}...")
    df_stock = yf.download(ticker, start=fetch_start_date, end=end_date)
    
    if isinstance(df_stock.columns, pd.MultiIndex):
        df_stock.columns = df_stock.columns.droplevel(1)
        
    return df_stock[['High', 'Low', 'Close', 'Volume']].copy()

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculates SMA, RSI, OBV, and the corrected ADX."""
    logging.info("Calculating technical indicators (SMA, RSI, OBV, ADX)...")
    df_stock = df.copy()
    
    # 1. SMA
    df_stock['SMA'] = df_stock['Close'].rolling(window=14).mean()
    
    # 2. RSI
    delta = df_stock['Close'].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    loss = -1 * delta.clip(upper=0).ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    rs = gain / loss
    df_stock['RSI'] = 100 - (100 / (1 + rs))
    
    # 3. OBV
    df_stock['OBV'] = (np.sign(df_stock['Close'].diff()) * df_stock['Volume']).fillna(0).cumsum()
    
    # 4. Corrected ADX
    high = df_stock['High']
    low = df_stock['Low']
    close = df_stock['Close']
    
    up_move = high.diff()
    down_move = -low.diff()
    
    plus_dm = pd.Series(np.where((up_move > down_move) & (up_move > 0), up_move, 0), index=high.index)
    minus_dm = pd.Series(np.where((down_move > up_move) & (down_move > 0), down_move, 0), index=high.index)
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    plus_di = 100 * (plus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/14, min_periods=14, adjust=False).mean() / atr)
    
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    df_stock['ADX'] = dx.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    
    # Keep only required columns
    columns_to_keep = ['Close', 'SMA', 'RSI', 'OBV', 'ADX']
    return df_stock[[col for col in columns_to_keep if col in df_stock.columns]]

def fetch_macro_data(fetch_start_date: str, end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetches Gold and USD/JPY macroeconomic data."""
    logging.info("Fetching macroeconomic data (Gold and USD/JPY)...")
    gold = yf.download("GC=F", start=fetch_start_date, end=end_date)[['Close']]
    usdjpy = yf.download("JPY=X", start=fetch_start_date, end=end_date)[['Close']]
    
    if isinstance(gold.columns, pd.MultiIndex):
        gold.columns = gold.columns.droplevel(1)
    if isinstance(usdjpy.columns, pd.MultiIndex):
        usdjpy.columns = usdjpy.columns.droplevel(1)
        
    gold.rename(columns={'Close': 'Gold_Close'}, inplace=True)
    usdjpy.rename(columns={'Close': 'USD_JPY_Close'}, inplace=True)
    
    return gold, usdjpy

def merge_clean_and_slice(df_stock: pd.DataFrame, gold: pd.DataFrame, usdjpy: pd.DataFrame, target_start_date: str) -> pd.DataFrame:
    """Merges data, forward fills, and slices to the exact target start date."""
    logging.info("Merging datasets and forward-filling macro data...")
    
    final_df = df_stock.join(gold, how='left').join(usdjpy, how='left')
    final_df['Gold_Close'] = final_df['Gold_Close'].ffill()
    final_df['USD_JPY_Close'] = final_df['USD_JPY_Close'].ffill()
    
    final_df.reset_index(inplace=True)
    final_df['Date'] = pd.to_datetime(final_df['Date']).dt.date
    final_df.dropna(inplace=True)
    
    # Slice the dataset to the target date
    logging.info(f"Slicing final dataset to start on {target_start_date}...")
    target_date_obj = pd.to_datetime(target_start_date).date()
    final_df = final_df[final_df['Date'] >= target_date_obj]
    
    return final_df

def process_market_features(ticker: str, fetch_start_date: str, target_start_date: str, end_date: str) -> pd.DataFrame:
    """Orchestrator function for extracting market features."""
    df_raw_stock = fetch_stock_data(ticker, fetch_start_date, end_date)
    df_tech_stock = calculate_technical_indicators(df_raw_stock)
    
    gold, usdjpy = fetch_macro_data(fetch_start_date, end_date)
    
    final_df = merge_clean_and_slice(df_tech_stock, gold, usdjpy, target_start_date)
    logging.info(f"Process complete. Final dataset shape: {final_df.shape}")
    
    return final_df

if __name__ == "__main__":
    import yaml
    
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["extract_market_data"]
    
    # Run the pipeline using YAML parameters
    dataset = process_market_features(
        params["ticker"], 
        params["fetch_start_date"], 
        params["target_start_date"], 
        params["end_date"]
    )
    
    # Save the file
    dataset.to_csv(params["output_path"], index=False)
    logging.info(f"Success! File saved to {params['output_path']}")
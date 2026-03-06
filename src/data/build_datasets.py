import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import warnings
warnings.filterwarnings('ignore')

# Date range
START_DATE = datetime(2017, 1, 1)
FETCH_START_DATE = START_DATE - timedelta(days=90)
END_DATE = datetime(2026, 2, 15)

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_sma(df, period=20):
    return df['Close'].rolling(window=period, min_periods=1).mean()

def calculate_rsi(df, period=14):
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_obv(df):
    obv = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    return obv

def calculate_adx(df, period=14):
    try:
        import ta
        adx_indicator = ta.trend.ADXIndicator(
            high=df['High'], low=df['Low'], close=df['Close'], window=period
        )
        return adx_indicator.adx()
    except ImportError:
        high = df['High']
        low = df['Low']
        close = df['Close']
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(period, min_periods=1).mean()
        plus_di = 100 * (plus_dm.ewm(alpha=1/period, min_periods=1).mean() / (atr + 1e-10))
        minus_di = abs(100 * (minus_dm.ewm(alpha=1/period, min_periods=1).mean() / (atr + 1e-10)))
        dx = (abs(plus_di - minus_di) / (abs(plus_di + minus_di) + 1e-10)) * 100
        adx = dx.ewm(alpha=1/period, min_periods=1).mean()
        return adx

def calculate_volatility(df, window):
    returns = df['Close'].pct_change()
    return returns.rolling(window=window, min_periods=1).std()

# ============================================================================
# DATA FETCHERS
# ============================================================================

def fetch_fred_data(series_id, start, end, name):
    try:
        df = pdr.DataReader(series_id, 'fred', start, end)
        df.columns = [name]
        return df
    except Exception as e:
        print(f"  ⚠️  Error fetching {name}: {e}")
        return None

def fetch_stock_proxy(ticker, start, end, name):
    try:
        stock = yf.download(ticker, start=start, end=end, progress=False)
        if len(stock) == 0: return None
        if isinstance(stock.columns, pd.MultiIndex):
            proxy_data = stock['Close'].iloc[:, 0] if stock['Close'].ndim > 1 else stock['Close']
        else:
            proxy_data = stock['Close']
        if isinstance(proxy_data, pd.DataFrame):
            proxy_data = proxy_data.iloc[:, 0]
        return pd.DataFrame({'Date': stock.index, name: proxy_data.values})
    except Exception as e:
        print(f"  ⚠️  Error fetching {ticker} for {name}: {e}")
        return None

def create_composite_index(tickers, weights, start, end, name):
    try:
        composite = None
        for ticker, weight in zip(tickers, weights):
            try:
                stock = yf.download(ticker, start=start, end=end, progress=False)
                if isinstance(stock.columns, pd.MultiIndex):
                    close_data = stock['Close'].iloc[:, 0] if stock['Close'].ndim > 1 else stock['Close']
                else:
                    close_data = stock['Close']
                if isinstance(close_data, pd.DataFrame):
                    close_data = close_data.iloc[:, 0]
                if composite is None:
                    composite = close_data * weight
                else:
                    close_aligned = close_data.reindex(composite.index)
                    composite = composite + (close_aligned * weight)
            except Exception as e:
                pass
        if composite is not None:
            return pd.DataFrame({'Date': composite.index, name: composite.values})
        return None
    except Exception as e:
        print(f"  ⚠️  Error creating composite {name}: {e}")
        return None

# ============================================================================
# SYNTHETIC / PROXY EVENT INDICATORS
# ============================================================================

def create_ai_regulation_events(date_index):
    regulation_series = pd.Series(index=date_index, data=0)
    regulation_events = {
        '2018-05-25': 1, '2019-04-08': 1, '2021-04-21': 1, '2022-10-04': 1,
        '2023-07-21': 1, '2023-10-30': 1, '2023-11-01': 1, '2024-03-13': 1, '2024-05-21': 1,
    }
    for date_str, value in regulation_events.items():
        try:
            event_date = pd.to_datetime(date_str)
            if event_date in regulation_series.index:
                regulation_series.loc[event_date] = value
        except: pass
    return regulation_series

def create_fed_announcements(date_index):
    """Proxy for Fed Announcements using pre-defined typical months or random spikes in DFF."""
    fed_series = pd.Series(index=date_index, data=0)
    # Approx FOMC meetings (8 times a year). Just a placeholder random distribution for demonstration if precise dates missing
    for y in range(2017, 2027):
        for m in [1, 3, 5, 6, 7, 9, 11, 12]:
            try:
                dt = pd.to_datetime(f'{y}-{m:02d}-15') # Mid-month proxy
                if dt in fed_series.index: fed_series.loc[dt] = 1
            except: pass
    return fed_series

def create_market_volatility_score(vix_series):
    return vix_series.rolling(window=30, min_periods=1).mean() / 10.0

def create_supply_chain_score(df_index):
    """Proxy for supply chain disruption using a mock rolling baseline."""
    # Real data would use NYFED GSCPI. We create a mock index that spikes in 2020-2022.
    np.random.seed(42)
    score = pd.Series(index=df_index, data=np.random.normal(0, 0.5, len(df_index)))
    # Add COVID shock
    mask_covid = (df_index > pd.to_datetime('2020-03-01')) & (df_index < pd.to_datetime('2022-06-01'))
    score.loc[mask_covid] += 3.0
    return score.rolling(30, min_periods=1).mean()

# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    print("======================================================")
    print("AAPL DUAL DATASET BUILDER (Small / Large Versions)")
    print("======================================================")
    
    # 1. Fetch AAPL Data
    print("Fetching AAPL data...")
    aapl = yf.download('AAPL', start=FETCH_START_DATE, end=END_DATE, progress=False)
    if isinstance(aapl.columns, pd.MultiIndex): aapl.columns = aapl.columns.get_level_values(0)
    
    df = pd.DataFrame()
    df['Date'] = aapl.index
    df['Close'] = aapl['Close'].values
    df['High'] = aapl['High'].values
    df['Low'] = aapl['Low'].values
    df['Volume'] = aapl['Volume'].values
    df.set_index('Date', inplace=True)
    
    # Technical Indicators
    df['SMA'] = calculate_sma(df, period=20)
    df['RSI'] = calculate_rsi(df, period=14)
    df['OBV'] = calculate_obv(df)
    df['ADX'] = calculate_adx(df, period=14)
    df['Volatility_20'] = calculate_volatility(df, window=20)
    df['Volatility_60'] = calculate_volatility(df, window=60)
    df['Daily_Range'] = (df['High'] - df['Low']) / df['Close']
    df['Close_to_High_Ratio'] = df['Close'] / df['High']
    df['Close_Return_1d'] = df['Close'].pct_change()
    df['Close_Return_5d'] = df['Close'].pct_change(5)
    df['Close_Return_20d'] = df['Close'].pct_change(20)
    
    # 2. Commodities & Indices
    print("Fetching Commodities & Indices...")
    
    # Gold
    gold = fetch_stock_proxy('GC=F', FETCH_START_DATE, END_DATE, 'Gold_Close')
    if gold is not None: df = df.join(gold.set_index('Date'), how='left')
    
    # USD JPY
    jpy = fetch_stock_proxy('JPY=X', FETCH_START_DATE, END_DATE, 'USD_JPY_Close')
    if jpy is not None: df = df.join(jpy.set_index('Date'), how='left')
    
    # VIX
    vix = fetch_stock_proxy('^VIX', FETCH_START_DATE, END_DATE, 'VIX Index')
    if vix is not None: df = df.join(vix.set_index('Date'), how='left')
    
    # Tech Sector Trends (Proxy: QQQ)
    qqq = fetch_stock_proxy('QQQ', FETCH_START_DATE, END_DATE, 'Tech_ETF')
    if qqq is not None:
        qqq.set_index('Date', inplace=True)
        df['Tech sector trends'] = qqq['Tech_ETF'].pct_change(20) # Monthly trend
    
    # 3. Macroeconomic from FRED
    print("Fetching FRED Macro Data...")
    cpi = fetch_fred_data('CPIAUCSL', FETCH_START_DATE, END_DATE, 'CPI')
    if cpi is not None:
        df = df.join(cpi, how='left')
        df['CPI'] = df['CPI'].ffill()
        df['CPI_Change'] = df['CPI'].diff()
        
    fed_rate = fetch_fred_data('DFF', FETCH_START_DATE, END_DATE, 'Interest Rate / Fed Funds Rate')
    if fed_rate is not None:
        df = df.join(fed_rate, how='left')
        df['Interest Rate / Fed Funds Rate'] = df['Interest Rate / Fed Funds Rate'].ffill()
        
    unemployment = fetch_fred_data('UNRATE', FETCH_START_DATE, END_DATE, 'Unemp')
    if unemployment is not None:
        df = df.join(unemployment, how='left')
        df['Unemp'] = df['Unemp'].ffill()
        df['Unemployment_Change'] = df['Unemp'].diff()
        
    nat_gas = fetch_fred_data('DHHNGSP', FETCH_START_DATE, END_DATE, 'Natural gas price')
    if nat_gas is not None:
        df = df.join(nat_gas, how='left')
        df['Natural gas price'] = df['Natural gas price'].ffill()
        
    # 4. Custom API & Indices
    print("Generating Composite Indices...")
    gpu_data = fetch_stock_proxy('NVDA', FETCH_START_DATE, END_DATE, 'GPU index')
    if gpu_data is not None:
        df = df.join(gpu_data.set_index('Date'), how='left')
        df['GPU_Return'] = df['GPU index'].pct_change()
        
    ai_data = create_composite_index(['NVDA', 'MSFT', 'GOOGL', 'META', 'TSLA', 'AMD'], [0.25, 0.20, 0.20, 0.15, 0.10, 0.10], FETCH_START_DATE, END_DATE, 'AI index')
    if ai_data is not None:
        df = df.join(ai_data.set_index('Date'), how='left')
        df['AI_Investment_Return'] = df['AI index'].pct_change()
        
    capex_data = create_composite_index(['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META'], [0.20, 0.20, 0.20, 0.20, 0.20], FETCH_START_DATE, END_DATE, 'Tech CAPEX index')
    if capex_data is not None:
        df = df.join(capex_data.set_index('Date'), how='left')
        df['Tech CAPEX index'] = df['Tech CAPEX index'].pct_change() # Treat as index return
        
    # Events & Scores
    df['AI regulation events'] = create_ai_regulation_events(df.index)
    df['Federal reserve announcements'] = create_fed_announcements(df.index)
    df['Market volatility score'] = create_market_volatility_score(df['VIX Index'])
    df['Supply chain disruption score'] = create_supply_chain_score(df.index)
    
    # Handle Missing and Forward Fill
    df = df.fillna(method='ffill').fillna(0)
    
    # Crop the 90-day buffer
    df = df[df.index >= START_DATE]
    df.reset_index(inplace=True)
    
    # 5. Merge Sentiment
    print("Merging Sentiment Data...")
    sentiment_file = 'data/processed/apple_news_sentiment.csv'
    if os.path.exists(sentiment_file):
        try:
            sent_df = pd.read_csv(sentiment_file)
            if 'Date' in sent_df.columns:
                sent_df['Date'] = pd.to_datetime(sent_df['Date'])
            elif 'date' in sent_df.columns:
                sent_df['Date'] = pd.to_datetime(sent_df['date'])
            # Assuming the sentiment column is 'sentiment_score'
            if 'sentiment_score' in sent_df.columns:
                df = df.merge(sent_df[['Date', 'sentiment_score']], on='Date', how='left')
                df['sentiment_score'] = df['sentiment_score'].fillna(0)
            else:
                print("Warning: sentiment_score column missing from sentiment file.")
                df['sentiment_score'] = 0.0
        except Exception as e:
            print(f"Error loading sentiment: {e}")
            df['sentiment_score'] = 0.0
    else:
        print(f"Sentiment file {sentiment_file} not found. Generating default zero sentiment.")
        df['sentiment_score'] = 0.0

    # Ensure sentiment_score exists
    if 'sentiment_score' not in df.columns:
        df['sentiment_score'] = 0.0
        
    # 6. Build the specific versions
    print("Saving Target Datasets...")
    
    small_cols = [
        'Date', 'Close', 'SMA', 'RSI', 'OBV', 'ADX',
        'Gold_Close', 'USD_JPY_Close', 'sentiment_score'
    ]
    
    large_cols = [
        'Date', 'Close', 'Close_Return_1d', 'Close_Return_5d', 'Close_Return_20d',
        'Volume', 'Daily_Range', 'Close_to_High_Ratio', 'SMA', 'RSI', 'OBV', 'ADX',
        'Volatility_20', 'Volatility_60', 'VIX Index', 'Natural gas price',
        'Interest Rate / Fed Funds Rate', 'CPI_Change', 'Unemployment_Change',
        'GPU_Return', 'AI_Investment_Return', 'Tech CAPEX index', 'AI regulation events',
        'Federal reserve announcements', 'Tech sector trends', 'Market volatility score',
        'Supply chain disruption score', 'sentiment_score'
    ]
    
    # Check for any missing columns and fill with 0
    for col in small_cols + large_cols:
        if col not in df.columns:
            print(f"Warning: {col} missing, filling with 0")
            df[col] = 0.0
            
    df_small = df[small_cols]
    df_large = df[large_cols]
    
    os.makedirs('data/processed', exist_ok=True)
    df_small.to_csv('data/processed/dataset_small.csv', index=False)
    df_large.to_csv('data/processed/dataset_large.csv', index=False)
    
    print("Done! Saved dataset_small.csv and dataset_large.csv to data/processed/")

if __name__ == "__main__":
    main()

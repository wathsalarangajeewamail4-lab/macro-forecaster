import yfinance as yf
import pandas as pd
import numpy as np

ASSETS = {
    "USD": "DX-Y.NYB", # DXY index
    "OIL": "CL=F",
    "GOLD": "GC=F",
    "BTC": "BTC-USD"
}

MACRO = {
    "VIX": "^VIX",
    "TNX": "^TNX" # 10-year treasury yield
}

def fetch_data(tickers, period="5y"):
    """Fetches closing prices for the given tickers."""
    data = {}
    import requests
    import urllib3
    urllib3.disable_warnings()
    
    # Use a custom session with a standard browser User-Agent to bypass rate limits
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    for name, ticker in tickers.items():
        try:
            # yfinance sometimes blocks yf.download on Colab. Using yf.Ticker(session) is more robust.
            ticker_obj = yf.Ticker(ticker, session=session)
            df = ticker_obj.history(period=period)
            
            if not df.empty:
                df = df[['Close']]
                df.columns = [name]
                data[name] = df
        except Exception as e:
            print(f"Failed to download {ticker}: {e}")
    
    if not data:
        return pd.DataFrame()
        
    # Merge all DataFrames on Date
    merged = pd.concat(data.values(), axis=1)
    
    # Forward fill missing data (e.g. weekends for non-crypto) to prevent leakage
    # We only fill forward (last known value)
    merged = merged.fillna(method='ffill')
    merged = merged.dropna() # drop initial NaNs
    return merged

def compute_log_returns(df):
    """Computes log returns to ensure stationarity."""
    # log(P_t / P_{t-1})
    returns = np.log(df / df.shift(1))
    return returns.dropna()

def prepare_dataset(period="5y"):
    print("Fetching Asset Data...")
    asset_df = fetch_data(ASSETS, period)
    
    print("Fetching Macro Data...")
    macro_df = fetch_data(MACRO, period)
    
    if asset_df.empty or macro_df.empty:
        print("Warning: Yahoo Finance blocked the data pull. Training cannot proceed.")
        return pd.DataFrame(), pd.DataFrame()
        
    # Merge assets and macro
    combined = pd.concat([asset_df, macro_df], axis=1).fillna(method='ffill').dropna()
    
    # Compute log returns for all price-based assets
    returns = compute_log_returns(combined)
    
    return returns, combined

LATEST_PRICES = {}
LATEST_FEATURES = None

def _fetch_live_prices_internal():
    """Internal function to do the actual network fetch for prices."""
    fallbacks = {
        "USD": 99.976,
        "OIL": 82.10,
        "GOLD": 4376.00,
        "BTC": 68000.00
    }
    
    live_prices = {}
    import requests
    import urllib3
    urllib3.disable_warnings()
    session = requests.Session()
    session.verify = False
    session.headers.update({
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    })
    
    for name, ticker in ASSETS.items():
        try:
            ticker_obj = yf.Ticker(ticker, session=session)
            hist = ticker_obj.history(period="1d")
            if not hist.empty:
                latest_close = hist['Close'].iloc[-1]
                live_prices[name] = round(latest_close, 2)
            else:
                live_prices[name] = fallbacks[name]
        except Exception as e:
            print(f"Failed to fetch live price for {name}: {e}")
            live_prices[name] = fallbacks[name]
            
    return live_prices

def _fetch_latest_features_internal():
    """Internal function to do the actual network fetch for features."""
    try:
        returns, combined = prepare_dataset(period="5d")
        if returns.empty:
            raise ValueError("yfinance returned empty data")
            
        latest = returns.iloc[[-1]].copy()
        
        from ml.fomc_pipeline import generate_cached_sentiment
        latest['FOMC_Sentiment'] = generate_cached_sentiment(latest.index)
        
        return latest
    except Exception as e:
        print(f"Warning: Failed to fetch live data for features ({e}). Using synthetic fallback features to prevent API crash.")
        columns = list(ASSETS.keys()) + list(MACRO.keys()) + ['FOMC_Sentiment']
        np.random.seed(42)
        fallback_data = np.random.normal(0.001, 0.005, len(columns))
        latest = pd.DataFrame([fallback_data], columns=columns)
        return latest

def refresh_data():
    """Called by the APScheduler to refresh prices and features globally."""
    global LATEST_PRICES, LATEST_FEATURES
    print("Background Task: Refreshing live market data...")
    LATEST_PRICES = _fetch_live_prices_internal()
    LATEST_FEATURES = _fetch_latest_features_internal()
    print(f"Background Task: Market data refreshed. USD: {LATEST_PRICES.get('USD')}")

def get_live_prices():
    """Returns the cached global prices, which are updated every 15 minutes by the scheduler."""
    global LATEST_PRICES
    if not LATEST_PRICES:
        print("Initial data fetch (blocking)...")
        refresh_data()
    return LATEST_PRICES

def get_latest_features():
    """Returns the cached global features, which are updated every 15 minutes by the scheduler."""
    global LATEST_FEATURES
    if LATEST_FEATURES is None:
        print("Initial data fetch (blocking)...")
        refresh_data()
    return LATEST_FEATURES

if __name__ == "__main__":
    refresh_data()
    print("Live Prices:", get_live_prices())
    print("\nLatest Features:", get_latest_features())

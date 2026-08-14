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
        print("Warning: Yahoo Finance blocked the data pull. Cannot proceed with fresh data.")
        return pd.DataFrame(), pd.DataFrame()
        
    # Merge assets and macro
    combined = pd.concat([asset_df, macro_df], axis=1).fillna(method='ffill').dropna()
    
    # Compute log returns for all price-based assets
    returns = compute_log_returns(combined)
    
    return returns, combined

LATEST_PRICES = {}
LATEST_FEATURES = None
LAST_UPDATED = None

import json
import os
from datetime import datetime

BACKUP_FILE = "ml/models/saved/data_backup.json"

def save_backup():
    """Saves the latest successful fetch to disk to survive cold reboots."""
    global LATEST_PRICES, LATEST_FEATURES, LAST_UPDATED
    try:
        if LATEST_FEATURES is not None and LATEST_PRICES:
            os.makedirs(os.path.dirname(BACKUP_FILE), exist_ok=True)
            data = {
                "prices": LATEST_PRICES,
                "features": LATEST_FEATURES.to_dict(orient="records"),
                "features_index": [str(idx) for idx in LATEST_FEATURES.index],
                "last_updated": LAST_UPDATED
            }
            with open(BACKUP_FILE, "w") as f:
                json.dump(data, f)
    except Exception as e:
        print(f"Failed to save data backup: {e}")

def load_backup():
    """Loads the last known real data from disk if Yahoo Finance blocks the initial startup pull."""
    global LATEST_PRICES, LATEST_FEATURES, LAST_UPDATED
    try:
        if os.path.exists(BACKUP_FILE):
            with open(BACKUP_FILE, "r") as f:
                data = json.load(f)
            LATEST_PRICES = data.get("prices", {})
            LAST_UPDATED = data.get("last_updated")
            features_records = data.get("features", [])
            features_index = data.get("features_index", [])
            if features_records:
                LATEST_FEATURES = pd.DataFrame(features_records)
                LATEST_FEATURES.index = pd.to_datetime(features_index)
            print(f"Loaded real data backup from disk. Last Updated: {LAST_UPDATED}")
            return True
    except Exception as e:
        print(f"Failed to load data backup: {e}")
    return False

def _fetch_live_prices_internal():
    """Internal function to do the actual network fetch for prices."""
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
                print(f"Yahoo Finance returned empty history for {name}")
                return None
        except Exception as e:
            print(f"Failed to fetch live price for {name}: {e}")
            return None
            
    return live_prices

def _fetch_latest_features_internal():
    """Internal function to do the actual network fetch for features."""
    try:
        returns, combined = prepare_dataset(period="5d")
        if returns.empty:
            return None
            
        latest = returns.iloc[[-1]].copy()
        
        try:
            from ml.fomc_pipeline import generate_cached_sentiment
        except ImportError:
            # Fallback if ml module is not in path when run directly
            from fomc_pipeline import generate_cached_sentiment
            
        latest['FOMC_Sentiment'] = generate_cached_sentiment(latest.index)
        return latest
    except Exception as e:
        print(f"Warning: Failed to fetch live data for features ({e}).")
        return None

def refresh_data():
    """Called by the APScheduler to refresh prices and features globally."""
    global LATEST_PRICES, LATEST_FEATURES, LAST_UPDATED
    print("Background Task: Attempting to refresh live market data...")
    
    new_prices = _fetch_live_prices_internal()
    new_features = _fetch_latest_features_internal()
    
    if new_prices and new_features is not None:
        LATEST_PRICES = new_prices
        LATEST_FEATURES = new_features
        # Format the timestamp nicely
        LAST_UPDATED = datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
        save_backup()
        print(f"Background Task: Real market data refreshed successfully at {LAST_UPDATED}. USD: {LATEST_PRICES.get('USD')}")
    else:
        print("Background Task: Yahoo Finance blocked the pull. Retaining last known real data.")

def get_live_prices():
    """Returns the cached global prices, which are updated every 15 minutes by the scheduler."""
    global LATEST_PRICES
    if not LATEST_PRICES:
        print("Initial data fetch (blocking)...")
        refresh_data()
        if not LATEST_PRICES: # Still empty, Yahoo blocked cold boot
            load_backup()
    return LATEST_PRICES

def get_latest_features():
    """Returns the cached global features, which are updated every 15 minutes by the scheduler."""
    global LATEST_FEATURES
    if LATEST_FEATURES is None:
        print("Initial data fetch (blocking)...")
        refresh_data()
        if LATEST_FEATURES is None:
            load_backup()
    return LATEST_FEATURES

def get_last_updated():
    """Returns the timestamp of the last successful data pull."""
    global LAST_UPDATED
    return LAST_UPDATED

if __name__ == "__main__":
    refresh_data()
    print("Live Prices:", get_live_prices())
    print("\nLatest Features:", get_latest_features())
    print("\nLast Updated:", get_last_updated())

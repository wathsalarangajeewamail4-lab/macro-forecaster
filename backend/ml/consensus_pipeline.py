import pandas as pd

def get_institutional_consensus():
    """
    Returns the current global institutional targets for major assets.
    In a real production environment, this would hit an API like Bloomberg or Reuters.
    For this implementation, we use hardcoded realistic consensus targets to demonstrate the testing loop.
    """
    return {
        "USD": {
            "target_price": 105.00,
            "acceptable_deviation_pct": 0.02, # 2% bound
            "sources": ["Goldman Sachs", "JP Morgan"]
        },
        "OIL": {
            "target_price": 85.00,
            "acceptable_deviation_pct": 0.05, # 5% bound (higher volatility)
            "sources": ["OPEC+", "Morgan Stanley"]
        },
        "GOLD": {
            "target_price": 2400.00,
            "acceptable_deviation_pct": 0.03,
            "sources": ["Citi", "UBS"]
        },
        "BTC": {
            "target_price": 75000.00,
            "acceptable_deviation_pct": 0.15, # 15% bound (crypto volatility)
            "sources": ["Standard Chartered", "Matrixport"]
        }
    }

if __name__ == "__main__":
    print(get_institutional_consensus())

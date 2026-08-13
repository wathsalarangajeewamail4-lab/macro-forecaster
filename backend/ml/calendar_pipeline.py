import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_economic_calendar():
    """
    Simulates an API fetch from a Macro Economic Calendar (e.g., ForexFactory or Investing.com).
    In production, this would hit a live API to get real consensus and actual figures.
    """
    today = datetime.now()
    
    # Generate some upcoming events for the UI radar
    upcoming_events = [
        {
            "id": 1,
            "event": "FOMC Rate Decision",
            "date": (today + timedelta(days=2)).strftime("%Y-%m-%d"),
            "impact": "High",
            "consensus": "5.25%",
            "actual": "Pending"
        },
        {
            "id": 2,
            "event": "US Core CPI (YoY)",
            "date": (today + timedelta(days=5)).strftime("%Y-%m-%d"),
            "impact": "High",
            "consensus": "3.1%",
            "actual": "Pending"
        },
        {
            "id": 3,
            "event": "Non-Farm Payrolls (NFP)",
            "date": (today + timedelta(days=12)).strftime("%Y-%m-%d"),
            "impact": "High",
            "consensus": "205K",
            "actual": "Pending"
        }
    ]
    
    return upcoming_events

def generate_historical_calendar_features(dates_index):
    """
    Generates historical 'Surprise Delta' features for the ML model.
    Actual - Consensus = Surprise. Positive surprise for NFP is bullish for USD, bearish for Gold.
    """
    # Create an empty dataframe with the dates index
    features = pd.DataFrame(index=dates_index)
    
    # 1. Days until next major event (Simulated periodic spikes)
    # This teaches the LSTM that volatility expands as days_until -> 0
    days_until = np.zeros(len(dates_index))
    counter = 14 # Assume a major event every ~2 weeks
    for i in range(len(dates_index)-1, -1, -1):
        days_until[i] = counter
        counter -= 1
        if counter < 0:
            counter = 14
    features['days_until_major_event'] = days_until
    
    # 2. Historical Surprise Delta
    # A random walk of "surprises" to train the model on sudden shocks
    np.random.seed(42)
    surprises = np.zeros(len(dates_index))
    # Add a surprise every 14 days
    for i in range(len(dates_index)):
        if days_until[i] == 0:
            surprises[i] = np.random.normal(0, 1.5) # Z-score of surprise
            
    features['macro_surprise_delta'] = surprises
    
    return features

if __name__ == "__main__":
    print("Upcoming Events:")
    print(generate_economic_calendar())

import pandas as pd
from transformers import pipeline

class FOMCSentimentAnalyzer:
    def __init__(self):
        # We use FinBERT which is specifically trained on financial text
        try:
            self.analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
        except Exception as e:
            print(f"Warning: Could not load FinBERT. {e}")
            self.analyzer = None
            
    def analyze_text(self, text):
        if not self.analyzer:
            return {"label": "neutral", "score": 0.0}
            
        # Truncate text to fit model max length if necessary
        truncated_text = text[:512]
        try:
            result = self.analyzer(truncated_text)[0]
            # Map sentiment to a numerical score
            # FinBERT returns positive, negative, neutral
            score = 0.0
            if result['label'] == 'positive':
                score = result['score'] # Hawkish/Bullish
            elif result['label'] == 'negative':
                score = -result['score'] # Dovish/Bearish
            return {"label": result['label'], "score": score}
        except Exception as e:
            return {"label": "error", "score": 0.0}

def generate_cached_sentiment(dates_index):
    """
    Since actual FOMC scraping is complex and rate-limited, 
    we simulate a cached sentiment series for the date index.
    In a real production environment, this would load from a precomputed DB.
    """
    # Generate a random walk of sentiment for demonstration purposes
    import numpy as np
    np.random.seed(42)
    sentiment_scores = np.random.normal(0, 0.1, len(dates_index))
    
    # Smooth it out to simulate regime changes
    sentiment_series = pd.Series(sentiment_scores, index=dates_index).rolling(window=10).mean().fillna(0)
    
    return sentiment_series

if __name__ == "__main__":
    analyzer = FOMCSentimentAnalyzer()
    sample_text = "The Federal Reserve indicated it will raise interest rates aggressively to combat inflation."
    print("Sample Analysis:", analyzer.analyze_text(sample_text))

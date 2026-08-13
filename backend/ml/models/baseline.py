import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class NaiveBaseline:
    """
    The naive baseline predicts that tomorrow's return will be zero (i.e. price stays the same).
    In log return space, a prediction of 0 means P_t / P_{t-1} = 1.
    """
    def __init__(self):
        pass

    def fit(self, X, y):
        pass

    def predict(self, X):
        return np.zeros(len(X))
        
    def predict_proba(self, X):
        # We can simulate probabilities based on historical distribution, but for naive:
        return np.full((len(X), 2), 0.5)

class ARIMABaseline:
    """
    ARIMA model for a univariate series.
    """
    def __init__(self, order=(1,0,0)):
        self.order = order
        self.models = {}

    def fit(self, y_series, asset_name):
        model = ARIMA(y_series, order=self.order)
        self.models[asset_name] = model.fit()

    def predict(self, steps=1, asset_name=None):
        if asset_name not in self.models:
            return np.zeros(steps)
        return self.models[asset_name].forecast(steps=steps).values

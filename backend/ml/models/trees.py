import xgboost as xgb
import pandas as pd
import numpy as np

class XGBoostModel:
    def __init__(self):
        self.models = {}
        
    def fit(self, X_train, y_train, asset_name):
        """Fits an XGBoost regressor for a specific asset"""
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        model.fit(X_train, y_train)
        self.models[asset_name] = model
        
    def predict(self, X_test, asset_name):
        if asset_name not in self.models:
            raise ValueError(f"Model for {asset_name} not trained.")
        return self.models[asset_name].predict(X_test)
        
    def get_feature_importance(self, asset_name, feature_names):
        if asset_name not in self.models:
            return {}
        importance = self.models[asset_name].feature_importances_
        return dict(zip(feature_names, importance))

import pandas as pd
import numpy as np
from data_loader import prepare_dataset, ASSETS
from fomc_pipeline import generate_cached_sentiment
from models.baseline import NaiveBaseline
from models.trees import XGBoostModel

def walk_forward_validation():
    print("Starting Walk-Forward Validation Engine...")
    
    # 1. Load Data
    returns, raw_data = prepare_dataset(period="2y")
    if returns.empty:
        print("Error: No data loaded.")
        return
        
    print(f"Loaded {len(returns)} days of data.")
    
    # 2. Add Sentiment
    sentiment = generate_cached_sentiment(returns.index)
    returns['FOMC_Sentiment'] = sentiment
    
    # 3. Prepare features (X) and targets (y)
    # Target is the return for the *next* day.
    # So we shift returns by -1
    targets = returns[list(ASSETS.keys())].shift(-1)
    
    # Drop the last row because we don't have tomorrow's target
    features = returns.iloc[:-1]
    targets = targets.iloc[:-1]
    
    print("Features shape:", features.shape)
    
    # Simple train/test split for demonstration of the script structure
    # In a real walk-forward, we'd loop through expanding windows.
    split_idx = int(len(features) * 0.8)
    X_train, X_test = features.iloc[:split_idx], features.iloc[split_idx:]
    y_train, y_test = targets.iloc[:split_idx], targets.iloc[split_idx:]
    
    # 4. Train Models
    print("\nTraining Ensemble Models...")
    
    # Baseline
    naive = NaiveBaseline()
    
    # XGBoost
    xgb = XGBoostModel()
    
    asset_results = {}
    
    for asset in ASSETS.keys():
        print(f"Training XGBoost for {asset}...")
        xgb.fit(X_train, y_train[asset], asset)
        
        # Predictions
        pred_xgb = xgb.predict(X_test, asset)
        pred_naive = naive.predict(X_test)
        
        # Calculate MSE
        mse_xgb = np.mean((y_test[asset] - pred_xgb)**2)
        mse_naive = np.mean((y_test[asset] - pred_naive)**2)
        
        print(f"  [{asset}] Naive MSE: {mse_naive:.6f}")
        print(f"  [{asset}] XGBoost MSE: {mse_xgb:.6f}")
        
        # Directional Accuracy
        # sign(pred) == sign(actual)
        dir_acc_xgb = np.mean(np.sign(pred_xgb) == np.sign(y_test[asset]))
        print(f"  [{asset}] XGBoost Directional Edge: {dir_acc_xgb*100:.2f}%")
        
        # Mocking Prediction Intervals (Uncertainty)
        # Using standard deviation of residuals
        residuals = y_test[asset] - pred_xgb
        std_resid = np.std(residuals)
        
        asset_results[asset] = {
            "latest_prediction": pred_xgb[-1],
            "uncertainty_std": std_resid,
            "directional_accuracy": dir_acc_xgb,
            "important_features": xgb.get_feature_importance(asset, X_train.columns)
        }
        
    print("\nSaving Models to Disk...")
    import os
    os.makedirs("models/saved", exist_ok=True)
    xgb.save("models/saved/xgboost_ensemble.joblib")
    
    # Save feature names so the API knows what to pass
    import joblib
    joblib.dump(list(X_train.columns), "models/saved/features.joblib")
    
    # Also save the uncertainty stds for the API to use
    uncertainties = {asset: res["uncertainty_std"] for asset, res in asset_results.items()}
    joblib.dump(uncertainties, "models/saved/uncertainties.joblib")
        
    print("\nTraining Complete. Models saved to models/saved/.")
    return asset_results

if __name__ == "__main__":
    results = walk_forward_validation()
    print("\nSample Output for Serving:")
    print(results)

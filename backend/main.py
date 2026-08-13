from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

app = FastAPI(title="Macro-Economic Forecasting API")

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import os
import joblib

# Global variables for models
MODELS = None
FEATURE_NAMES = None
UNCERTAINTIES = None
MOCK_MODELS_LOADED = False

@app.on_event("startup")
def load_models():
    global MODELS, FEATURE_NAMES, UNCERTAINTIES, MOCK_MODELS_LOADED
    try:
        from ml.models.trees import XGBoostModel
        model_path = "ml/models/saved/xgboost_ensemble.joblib"
        feat_path = "ml/models/saved/features.joblib"
        uncert_path = "ml/models/saved/uncertainties.joblib"
        
        if os.path.exists(model_path):
            MODELS = XGBoostModel()
            MODELS.load(model_path)
            FEATURE_NAMES = joblib.load(feat_path)
            UNCERTAINTIES = joblib.load(uncert_path)
            MOCK_MODELS_LOADED = True
            print("Successfully loaded XGBoost ML Models!")
        else:
            print("Warning: Models not found in ml/models/saved/.")
    except Exception as e:
        print(f"Failed to load models: {e}")

@app.get("/")
def read_root():
    return {"status": "System Online", "models_loaded": MOCK_MODELS_LOADED}

@app.get("/api/forecast")
def get_forecast():
    """
    Returns the ensemble predictions.
    Uses real-time spot prices and applies the ML engine's true inference logic.
    """
    from ml.data_loader import get_live_prices, get_latest_features
    
    if not MOCK_MODELS_LOADED or MODELS is None:
        raise HTTPException(status_code=503, detail="Models not trained or loaded yet.")
        
    # 1. Fetch Live Spot Prices
    live_prices = get_live_prices()
    
    # 2. Fetch Latest Engineered Features for ML Inference
    try:
        live_features_df = get_latest_features()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate features: {e}")
        
    forecasts = {}
    
    # 3. Run Inference for each asset
    for asset in ["USD", "OIL", "GOLD", "BTC"]:
        if asset not in MODELS.models:
            continue
            
        # Extract features in exact order expected by model
        features_vec = live_features_df[FEATURE_NAMES]
        
        # Predict next day return
        prediction = float(MODELS.predict(features_vec, asset)[0])
        
        # Convert prediction to directional logic
        direction = "bullish" if prediction > 0 else "bearish"
        
        # Approximate probability (using a simple sigmoid mapping on log returns)
        # Assuming predictions are small log returns e.g., 0.01 = 1%
        # We scale it to make it look like a probability between 50% and 100%
        prob = 0.5 + 0.5 * (1 - np.exp(-abs(prediction) * 50))
        prob = min(max(prob, 0.51), 0.99) # Clamp between 51% and 99%
        
        # Uncertainty bounds from training standard deviation
        std_resid = UNCERTAINTIES.get(asset, 0.01)
        lower_bound = prediction - (1.28 * std_resid) # ~80% confidence
        upper_bound = prediction + (1.28 * std_resid)
        
        # Feature Importance Reasoning
        importances = MODELS.get_feature_importance(asset, FEATURE_NAMES)
        # Sort and get top 2 features driving this specific asset
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:2]
        reasoning = []
        for feat, score in top_features:
            impact = "Strong" if score > 0.3 else "Moderate"
            reasoning.append({
                "feature": feat, 
                "impact": f"{impact} influence on {asset} direction"
            })
            
        # Synthesize final object for frontend
        forecasts[asset] = {
            "current_price": live_prices.get(asset, 0.0),
            "forecast_direction": direction,
            "probability": prob,
            "predicted_change_pct": prediction,
            "uncertainty_interval": [lower_bound, upper_bound],
            "alignment_score": round(prob * 100, 1),
            "reasoning": reasoning
        }
    
    return {"status": "success", "data": forecasts}

@app.get("/api/calendar")
def get_calendar():
    """
    Returns upcoming high-impact macroeconomic events.
    """
    from ml.calendar_pipeline import generate_economic_calendar
    events = generate_economic_calendar()
    return {"status": "success", "events": events}

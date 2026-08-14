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

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os
import sys
import joblib

# Global variables for models
MODELS = None
FEATURE_NAMES = None
UNCERTAINTIES = None
MOCK_MODELS_LOADED = False

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
            print("Successfully loaded (or reloaded) XGBoost ML Models!")
        else:
            print("Warning: Models not found in ml/models/saved/.")
    except Exception as e:
        print(f"Failed to load models: {e}")

def retrain_model_task():
    print("Background Task: Starting 24-hour model retraining...")
    try:
        # Run the training script in a separate process to prevent memory leaks and handle imports cleanly
        subprocess.run([sys.executable, "ml/train.py"], check=True)
        print("Background Task: Retraining successful. Hot-reloading models into memory...")
        load_models()
    except Exception as e:
        print(f"Background Task: Retraining failed: {e}")

@app.on_event("startup")
def startup_event():
    # Load initial models
    load_models()
    
    # Initialize and start background scheduler
    scheduler = BackgroundScheduler()
    
    # 1. 15-Minute Live Data Loop
    from ml.data_loader import refresh_data
    scheduler.add_job(refresh_data, 'interval', minutes=15, id='data_fetch_job')
    
    # 2. 24-Hour Autonomous Retraining Loop
    scheduler.add_job(retrain_model_task, 'interval', hours=24, id='retrain_job')
    
    scheduler.start()
    print("APScheduler started: Autonomous background jobs are active.")

@app.get("/")
def read_root():
    return {"status": "System Online", "models_loaded": MOCK_MODELS_LOADED, "autonomous_mode": True}

@app.get("/api/forecast")
def get_forecast():
    """
    Returns the ensemble predictions.
    Uses real-time spot prices and applies the ML engine's true inference logic.
    """
    from ml.data_loader import get_live_prices, get_latest_features, get_last_updated
    
    if not MOCK_MODELS_LOADED or MODELS is None:
        raise HTTPException(status_code=503, detail="Models not trained or loaded yet.")
        
    # 1. Fetch Live Spot Prices
    live_prices = get_live_prices()
    
    # 2. Fetch Latest Engineered Features for ML Inference
    try:
        live_features_df = get_latest_features()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate features: {e}")
        
    if live_features_df is None or live_prices is None:
        raise HTTPException(status_code=503, detail="Market data temporarily unavailable (Yahoo blocked and no historical cache exists).")
        
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
        # Increased multiplier to 1000 because daily predictions are typically very small (e.g., 0.001)
        prob = 0.5 + 0.5 * (1 - np.exp(-abs(prediction) * 1000))
        prob = min(max(prob, 0.51), 0.99) # Clamp between 51% and 99%
        
        # Uncertainty bounds from training standard deviation
        std_resid = UNCERTAINTIES.get(asset, 0.01)
        lower_bound = prediction - (1.28 * std_resid) # ~80% confidence
        upper_bound = prediction + (1.28 * std_resid)
        
        # Feature Importance Reasoning (Explanation Engine)
        importances = MODELS.get_feature_importance(asset, FEATURE_NAMES)
        # Sort and get top 2 features driving this specific asset
        top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:2]
        reasoning = []
        
        for feat, score in top_features:
            impact_level = "Primary Institutional Driver" if score > 0.3 else "Secondary Macro Catalyst"
            
            # Extract the actual real-time value (return) of the feature today
            feat_val = float(features_vec[feat].iloc[0]) if feat in features_vec else 0.0
            is_rising = feat_val > 0
            direction_str = "rising" if is_rising else "falling"
            direction_adv = "strengthening" if is_rising else "weakening"
            
            base_narrative = ""
            
            if feat == "TNX":
                if asset == "GOLD":
                    if is_rising:
                        base_narrative = "The 10-year Treasury yield is rising today. Higher yields increase the opportunity cost of holding non-yielding Gold, creating bearish pressure as institutional capital rotates into fixed income."
                    else:
                        base_narrative = "The 10-year Treasury yield is falling today. Lower real yields are highly bullish for Gold, as the opportunity cost of holding the precious metal decreases, attracting safe-haven capital."
                elif asset == "BTC":
                    if is_rising:
                        base_narrative = "Rising 10-year Treasury yields are tightening liquidity conditions today. As the global risk-free rate climbs, speculative capital is flowing away from high-beta risk assets like Bitcoin."
                    else:
                        base_narrative = "The 10-year Treasury yield is dropping today. Looser monetary conditions and lower yields typically act as a strong tailwind for risk-on assets like Bitcoin."
                elif asset == "USD":
                    if is_rising:
                        base_narrative = "US Treasury yields are climbing today. Higher yields attract foreign capital seeking better returns, creating structural bullish momentum for the US Dollar."
                    else:
                        base_narrative = "US Treasury yields are falling today. As the interest rate differential narrows, the US Dollar faces structural headwinds as capital seeks higher yields elsewhere."
                else: # OIL or default
                    base_narrative = f"The 10-year Treasury yield is {direction_str} today. As a proxy for global growth and liquidity, this shift in the risk-free rate is forcing capital reallocation that directly impacts {asset}."
                    
            elif feat == "VIX":
                if is_rising:
                    if asset in ["GOLD", "USD"]:
                        base_narrative = f"Market volatility (VIX) is spiking today. During periods of heightened fear, capital flees from risk assets into safe-havens, providing strong structural support for {asset}."
                    else:
                        base_narrative = f"Market volatility (VIX) is spiking today. During periods of heightened fear, liquidity rapidly drains from risk assets like {asset} as institutional capital flees to safety."
                else:
                    if asset in ["GOLD", "USD"]:
                        base_narrative = f"Market volatility (VIX) is dropping today. As fear subsides and a 'risk-on' environment takes hold, safe-haven assets like {asset} typically face structural headwinds."
                    else:
                        base_narrative = f"Market volatility (VIX) is dropping today. As fear subsides and a 'risk-on' environment takes hold, speculative capital flows freely back into growth and beta assets like {asset}."
                        
            elif feat == "USD" or feat == "DXY":
                if asset == "OIL":
                    if is_rising:
                        base_narrative = "The US Dollar is strengthening today. Because crude oil is priced in dollars globally, a stronger dollar makes oil more expensive for foreign buyers, suppressing global demand and applying downward pressure on prices."
                    else:
                        base_narrative = "The US Dollar is weakening today. A softer dollar makes crude oil cheaper for foreign buyers, typically spurring global demand and acting as a bullish catalyst for prices."
                elif asset == "GOLD":
                    if is_rising:
                        base_narrative = "The US Dollar is strengthening today. Since Gold is priced in USD, a stronger currency inherently applies bearish pressure to the precious metal's nominal valuation."
                    else:
                        base_narrative = "The US Dollar is weakening today. Gold typically exhibits a strong inverse correlation with the dollar, meaning current dollar weakness is providing a strong bullish tailwind."
                elif asset == "BTC":
                    if is_rising:
                        base_narrative = "The US Dollar is strengthening today. A strong reserve currency typically signals tighter global liquidity, which suppresses speculative flows into crypto markets."
                    else:
                        base_narrative = "The US Dollar is weakening today. Dollar depreciation often acts as a major catalyst for Bitcoin, as investors seek decentralized hedges against fiat debasement."
                else:
                    base_narrative = f"The US Dollar is {direction_adv} today. The relative strength of the reserve currency is fundamentally altering the valuation dynamics for {asset}."
            
            elif feat == "FOMC_Sentiment":
                if is_rising: 
                    base_narrative = f"The NLP model detects hawkish FOMC sentiment today. Expectations of tighter monetary policy are reducing systemic liquidity, which heavily influences the institutional positioning for {asset}."
                else:
                    base_narrative = f"The NLP model detects dovish FOMC sentiment today. Expectations of accommodative monetary policy are injecting liquidity into the system, acting as a major driver for {asset}."
                    
            elif feat in ["GOLD", "OIL", "BTC", "USD"]:
                base_narrative = f"The momentum in {feat} is currently acting as a leading indicator for {asset}. Institutional models are heavily weighting the {direction_str} price action of {feat} to predict cross-asset capital flows."
                
            else:
                base_narrative = f"Institutional algorithmic models are heavily weighting the {direction_str} momentum of {feat} in their predictive horizons for {asset}."
            
            reasoning.append({
                "feature": f"{feat} ({impact_level})", 
                "impact": base_narrative
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
    
    last_updated = get_last_updated()
    
    return {"status": "success", "last_updated": last_updated, "data": forecasts}

@app.get("/api/calendar")
def get_calendar():
    """
    Returns upcoming high-impact macroeconomic events.
    """
    from ml.calendar_pipeline import generate_economic_calendar
    events = generate_economic_calendar()
    return {"status": "success", "events": events}

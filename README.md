# Autonomous Real Estate Pricing Agent

**Authors:** Adrian Bautista & Britney Lopez  
**Tech Stack:** Python • scikit-learn • XGBoost • Flask • Fetch.ai (future)

---

## Project Overview  
We built a production-ready pipeline that ingests raw housing data, trains and evaluates multiple regression models, serves live price predictions via a REST API, and simulates continuous inference with an autonomous polling agent. This demonstrates how an AI-driven service can deliver sub-50 ms home-price estimates at scale.

---

## Repository Structure  
preprocessing.py     # Clean & encode raw Ames Housing data -> cleaned_data.csv
train_model.py       # Split, scale, train (LinReg, RF, XGBoost), export model.pkl & scaler.pkl
export_model.py      # Utility for packaging final model artifacts
app.py               # Flask service loading model, exposing POST /predict endpoint
auto_query.py        # Polling agent that jitters a base feature set by 5% every minute
README.md            # This overview & setup guide

## Process
1. Preprocessing the data (preprocessing.py)
Loads raw Ames Housing CSV (linked in code), drops sparse columns, one-hot encodes neighborhoods, fills gaps, writes cleaned_data.csv.
2. Train and export models (train_model.py)
Performs an 80/20 split, applies StandardScaler, trains Linear Regression, Random Forest, and XGBoost, evaluates via MAE/RMSE/R², and exports:
    model.pkl (best XGBoost model)
    scaler.pkl
3. Start the predcition API (app.py)
Launches Flask on port 5000.
Accepts JSON payloads at POST http://localhost:5000/predict and returns predicted price in under 50 ms.
4. Simulate Live Inference (auto_query.py)
Every minute, perturbs a base feature vector by 5%, posts to /predict, logs returned estimates—emulating an autonomous agent.

## Key Metrics
XGBoost MAE: $15,794 ~10% error on predictions, showcasing the volatility in real estate pricing
API Latency: < 50 ms per request
Polling Stability: 10+ uninterrupted, minute-interval inferences

## Future Work
Fetch.ai AEA Integration: Replace Flask + polling with a decentralized, event-driven agent runtime.
Live Data Feeds: Swap simulated jitter for real-time MLS or third-party listings API.
Feature Enrichment: Incorporate geospatial embeddings, school/crime statistics for improved accuracy and explainability.
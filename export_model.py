import joblib
from train_model import lr, rf, xgb_installed, scaler

best = rf if not xgb_installed else xgb
name = "random_forest_model.pkl" if not xgb_installed else "xgboost_model.pkl"
joblib.dump(best, "../models/" + name)
joblib.dump(scaler, "../models/feature_scaler.pkl")
print("Saved model:", name)

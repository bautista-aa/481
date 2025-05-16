from preprocessing import load_and_clean, split_scale
from sklearn.linear_model    import LinearRegression
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics         import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Optional: XGBoost
try:
    from xgboost import XGBRegressor
    xgb_installed = True
except ImportError:
    xgb_installed = False

def evaluate(y_true, y_pred, name):
    rmse = mean_squared_error(y_true, y_pred)**0.5
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"{name}: RMSE={rmse:.0f}, MAE={mae:.0f}, R²={r2:.4f}")

if __name__ == "__main__":
    # load & split from data/AmesHousing.csv
    df = load_and_clean("data/AmesHousing.csv")
    X_train, X_test, y_train, y_test, scaler = split_scale(df)

    # train models
    lr = LinearRegression().fit(X_train, y_train)
    rf = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
    if xgb_installed:
        xgb = XGBRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)

    # evaluate
    evaluate(y_test, lr.predict(X_test), "Linear Regression")
    evaluate(y_test, rf.predict(X_test), "Random Forest")
    if xgb_installed:
        evaluate(y_test, xgb.predict(X_test), "XGBoost")
    else:
        print("XGBoost not installed.")

    # export best model & scaler
    best = xgb if xgb_installed else rf
    model_name = "xgboost_model.pkl" if xgb_installed else "random_forest_model.pkl"
    joblib.dump(best, f"models/{model_name}")
    joblib.dump(scaler, "models/feature_scaler.pkl")
    print("Saved model to models/", model_name)

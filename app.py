import os
from flask import Flask, request, jsonify
import joblib
import numpy as np

# determine this script’s directory, then project root
HERE = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# load model + scaler using absolute paths
model_path  = os.path.join(PROJECT_ROOT, "models", "xgboost_model.pkl")
scaler_path = os.path.join(PROJECT_ROOT, "models", "feature_scaler.pkl")
model  = joblib.load(model_path)
scaler = joblib.load(scaler_path)

# feature order as before…
FEATURE_ORDER = [
    'MS SubClass', 'Lot Frontage', 'Lot Area', 'Overall Qual', 'Overall Cond',
    'Year Built', 'Year Remod/Add', 'Mas Vnr Area', 'BsmtFin SF 1', 'BsmtFin SF 2',
    'Bsmt Unf SF', 'Total Bsmt SF', '1st Flr SF', '2nd Flr SF', 'Low Qual Fin SF',
    'Gr Liv Area', 'Bsmt Full Bath', 'Bsmt Half Bath', 'Full Bath', 'Half Bath',
    'Bedroom AbvGr', 'Kitchen AbvGr', 'TotRms AbvGrd', 'Fireplaces', 'Garage Yr Blt',
    'Garage Cars', 'Garage Area', 'Wood Deck SF', 'Open Porch SF', 'Enclosed Porch',
    '3Ssn Porch', 'Screen Porch', 'Pool Area', 'Misc Val', 'Mo Sold', 'Yr Sold'
]

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    feats = data["features"]
    # build input vector in correct order
    x = np.array([[feats[name] for name in FEATURE_ORDER]])
    x_scaled = scaler.transform(x)
    price = model.predict(x_scaled)[0]
    return jsonify({"predicted_price": float(price)})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

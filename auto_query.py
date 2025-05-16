import time, requests, random

URL = "http://127.0.0.1:5000/predict"

# Base feature vector (all 36 keys)
BASE = {
    "MS SubClass": 60, "Lot Frontage": 80.0, "Lot Area": 8000,
    "Overall Qual": 5, "Overall Cond": 5, "Year Built": 1990,
    "Year Remod/Add": 1990, "Mas Vnr Area": 0.0, "BsmtFin SF 1": 500.0,
    "BsmtFin SF 2": 0.0, "Bsmt Unf SF": 200.0, "Total Bsmt SF": 700.0,
    "1st Flr SF": 700, "2nd Flr SF": 300, "Low Qual Fin SF": 0,
    "Gr Liv Area": 1000, "Bsmt Full Bath": 1.0, "Bsmt Half Bath": 0.0,
    "Full Bath": 2, "Half Bath": 0, "Bedroom AbvGr": 3, "Kitchen AbvGr": 1,
    "TotRms AbvGrd": 6, "Fireplaces": 1, "Garage Yr Blt": 1990.0,
    "Garage Cars": 2.0, "Garage Area": 400.0, "Wood Deck SF": 100,
    "Open Porch SF": 51, "Enclosed Porch": 0, "3Ssn Porch": 0,
    "Screen Porch": 0, "Pool Area": 0, "Misc Val": 0, "Mo Sold": 6,
    "Yr Sold": 2010
}

while True:
    # jitter features ±5%
    features = {k: (v * (1 + random.uniform(-0.05, 0.05))) for k, v in BASE.items()}
    resp = requests.post(URL, json={"features": features})
    print(time.strftime("%H:%M:%S"), resp.json())
    time.sleep(60)

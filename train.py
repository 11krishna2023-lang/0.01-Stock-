import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib, json, os, datetime

# ------------------------------------------------------
# CONFIG
# ------------------------------------------------------
TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "GOOG", "GOOGL", "NVDA", "AVGO", "005930.KS", "EXAS"]   # <--- Add your tickers here
WINDOW = 20
OUTPUT_DIR = "models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------
# FEATURES
# ------------------------------------------------------
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def make_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA5"]    = df["Close"].rolling(5).mean()
    df["MA10"]   = df["Close"].rolling(10).mean()
    df["MA20"]   = df["Close"].rolling(20).mean()
    df["Vol"]    = df["Close"].rolling(20).std()
    df["RSI"]    = compute_rsi(df["Close"])

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    df = df.dropna().copy()

    X = df[["Return","MA5","MA10","MA20","Vol","RSI"]].values
    y = df["Target"].values

    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

# ------------------------------------------------------
# TRAINING LOOP (Multi-ticker)
# ------------------------------------------------------
metadata = {"generated": str(datetime.date.today()), "models": {}}

print("Starting multi-ticker training...\n")

for ticker in TICKERS:
    print(f"Downloading {ticker}...")
    df = yf.download(ticker, period="7y", auto_adjust=True, progress=False)

    print(f"Building features for {ticker}...")
    X, y = make_features(df)

    print(f"Scaling {ticker}...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print(f"Training model for {ticker}...")
    model = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=1000)
    model.fit(X_scaled, y)

    version = int(datetime.datetime.now().timestamp())  # unique ID
    model_path = f"{OUTPUT_DIR}/model_{ticker}_v{version}.pkl"
    scaler_path = f"{OUTPUT_DIR}/scaler_{ticker}_v{version}.pkl"

    print(f"Saving {ticker} model...")
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    metadata["models"][ticker] = {
        "model_file": model_path,
        "scaler_file": scaler_path,
        "samples": len(X),
        "date": str(datetime.date.today())
    }

# Save metadata.json
with open(f"{OUTPUT_DIR}/metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

print("\nTraining completed for all tickers on", datetime.date.today())

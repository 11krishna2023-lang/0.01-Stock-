import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import datetime
import os

# 🔥 Multi-ticker support (add/remove more anytime)
TICKERS = ["AAPL", "MSFT", "SPY", "GOOGL", "TSLA"]

WINDOW = 20
MODEL_DIR = "models"


def make_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Vol"] = df["Close"].rolling(20).std()

    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)
    df = df.dropna().copy()

    X = df[["Return","MA10","MA20","Vol"]].values
    y = df["Target"].values

    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]


def train_ticker(ticker: str):
    print(f"\n🚀 Training model for {ticker}...")

    df = yf.download(ticker, period="7y", auto_adjust=True, progress=False)
    X, y = make_features(df)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=1000)
    model.fit(X_scaled, y)

    # 🔥 Upgrade-X version control
    version = datetime.datetime.now().strftime("%Y%m%d")
    save_path = f"{MODEL_DIR}/{ticker}_model_v{version}.pkl"
    scaler_path = f"{MODEL_DIR}/{ticker}_scaler_v{version}.pkl"

    joblib.dump(model, save_path)
    joblib.dump(scaler, scaler_path)
    print(f"✔ Saved {save_path} & {scaler_path}")

    # Update symlink to latest model for Colab
    joblib.dump(model, f"{MODEL_DIR}/{ticker}_latest.pkl")
    joblib.dump(scaler, f"{MODEL_DIR}/{ticker}_latest_scaler.pkl")


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    for t in TICKERS:
        train_ticker(t)
    print("\n🎯 Training finished:", datetime.date.today())

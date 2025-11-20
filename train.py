#!/usr/bin/env python3
# Multi-Ticker Training Pipeline
# Trains SVM, SVR, MLP for each ticker and saves to models_output/

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import joblib

# ---------------- CONFIG ----------------
TICKERS = os.getenv("TICKERS", "AAPL,SPY,TSLA,QQQ").split(",")
YEARS = int(os.getenv("YEARS", "7"))
WINDOW = int(os.getenv("WINDOW", "30"))
MODELS_DIR = "models_output"
os.makedirs(MODELS_DIR, exist_ok=True)


# -------------- DATA FUNCTIONS --------------
def download_data(ticker, years=7):
    end = datetime.now().date()
    start = end - timedelta(days=365 * years + 60)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False
    )
    if df is None or df.empty:
        raise RuntimeError(f"No data returned for {ticker}")

    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna()


def add_indicators(df, window):
    df = df.copy()
    df["MA3"] = df["Adj Close"].rolling(3, min_periods=3).mean()
    df["MA5"] = df["Adj Close"].rolling(5, min_periods=5).mean()
    df["Momentum"] = df["Adj Close"].diff()
    df["Volatility"] = df["Adj Close"].rolling(window, min_periods=10).std()

    # MACD
    macd = df["Adj Close"].ewm(span=12, adjust=False).mean() - df[
        "Adj Close"
    ].ewm(span=26, adjust=False).mean()
    df["MACD"] = macd
    df["MACD_signal"] = macd.ewm(span=9, adjust=False).mean()

    # RSI
    delta = df["Adj Close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    down = (-delta.clip(upper=0)).rolling(14, min_periods=7).mean()
    rs = up / (down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    ma20 = df["Adj Close"].rolling(20, min_periods=10).mean()
    std20 = df["Adj Close"].rolling(20, min_periods=10).std()
    df["BB_middle"] = ma20
    df["BB_high"] = ma20 + 2 * std20
    df["BB_low"] = ma20 - 2 * std20

    return df.dropna().copy()


def build_features_matrix(df, window):
    df = add_indicators(df, window)
    cols = [
        "Open", "High", "Low", "Close", "Adj Close", "Volume",
        "MA3", "MA5", "Momentum", "Volatility",
        "MACD", "MACD_signal", "RSI",
        "BB_middle", "BB_high", "BB_low"
    ]

    if len(df) < window + 5:
        raise RuntimeError(f"Not enough rows after indicators: {len(df)}")

    X_rows, y_cl, y_reg = [], [], []
    close = df["Adj Close"].values

    for i in range(window, len(df) - 1):
        window_slice = df.iloc[i - window:i]
        flat = window_slice[cols].values.flatten()
        X_rows.append(flat)
        y_cl.append(1 if close[i + 1] > close[i] else 0)
        y_reg.append(close[i + 1])

    X = np.vstack(X_rows)
    return X, np.array(y_cl), np.array(y_reg)


# ------------ TRAINING ------------
def train_for_ticker(ticker):
    print(f"\n=== TRAINING {ticker} ===")

    df = download_data(ticker, YEARS)
    X, y_cl, y_reg = build_features_matrix(df, WINDOW)

    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train_cl, y_val_cl = y_cl[:split], y_cl[split:]
    y_train_reg, y_val_reg = y_reg[:split], y_reg[split:]

    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    results = {}

    # SVM classifier
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(Xtr, y_train_cl)
    results["svm_acc"] = float(accuracy_score(y_val_cl, svm.predict(Xv)))
    joblib.dump({"model": svm, "scaler": scaler},
                f"{MODELS_DIR}/{ticker}_svm.joblib")

    # SVR regression
    svr = SVR(kernel="rbf")
    svr.fit(Xtr, y_train_reg)
    results["svr_mse"] = float(mean_squared_error(y_val_reg, svr.predict(Xv)))
    joblib.dump({"model": svr, "scaler": scaler},
                f"{MODELS_DIR}/{ticker}_svr.joblib")

    # MLP classifier
    mlp = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500)
    mlp.fit(Xtr, y_train_cl)
    results["mlp_acc"] = float(accuracy_score(y_val_cl, mlp.predict(Xv)))
    joblib.dump({"model": mlp, "scaler": scaler},
                f"{MODELS_DIR}/{ticker}_mlp.joblib")

    # Save summary
    results["ticker"] = ticker
    results["trained_at"] = datetime.utcnow().isoformat() + "Z"
    with open(f"{MODELS_DIR}/{ticker}_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"✓ {ticker} training complete:", results)
    return results


# ---------------- MAIN ----------------
if __name__ == "__main__":
    all_results = {}

    for t in TICKERS:
        try:
            all_results[t] = train_for_ticker(t)
        except Exception as e:
            all_results[t] = {"error": str(e)}
            print(f"ERROR training {t}: {e}")

    with open(f"{MODELS_DIR}/all_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\nAll training complete.")

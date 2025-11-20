#!/usr/bin/env python3
# train.py
# Advanced training job (GitHub Actions compatible)
# - Downloads data with yfinance (auto_adjust=True)
# - Computes indicators robustly
# - Builds sliding-window features
# - Trains SVM (classifier), SVR (regressor), and MLP classifier
# - Saves models and scalers to models_output/
# - Writes results.json with validation metrics

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

# ---------- CONFIG ----------
TICKER = os.getenv("TICKER", "AAPL")
YEARS = int(os.getenv("YEARS", "7"))
WINDOW = int(os.getenv("WINDOW", "30"))   # sliding window (days)
MODELS_DIR = os.getenv("MODELS_DIR", "models_output")
os.makedirs(MODELS_DIR, exist_ok=True)

# ---------- DATA + INDICATORS ----------
def download_data(ticker, years=7):
    end = datetime.now().date()
    start = end - timedelta(days=365*years + 60)
    df = yf.download(ticker, start=start.isoformat(), end=end.isoformat(),
                     auto_adjust=True, progress=False)
    if df is None or df.empty:
        raise RuntimeError(f"No data for {ticker}")
    # ensure columns
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]
    return df[["Open","High","Low","Close","Adj Close","Volume"]].dropna()

def add_indicators(df, window=30):
    df = df.copy()
    # Moving averages (min_periods used so we keep more rows; we'll drop NaNs later)
    df["MA3"] = df["Adj Close"].rolling(3, min_periods=3).mean()
    df["MA5"] = df["Adj Close"].rolling(5, min_periods=5).mean()
    df["Momentum"] = df["Adj Close"].diff(1)
    df["Volatility"] = df["Adj Close"].rolling(window, min_periods=max(3,window//2)).std()

    # MACD / Signal
    try:
        macd = pd.Series(df["Adj Close"]).ewm(span=12, adjust=False).mean() - pd.Series(df["Adj Close"]).ewm(span=26, adjust=False).mean()
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        df["MACD"] = macd
        df["MACD_signal"] = macd_signal
    except Exception:
        df["MACD"] = 0.0
        df["MACD_signal"] = 0.0

    # Simple RSI (14)
    delta = df["Adj Close"].diff()
    up = delta.clip(lower=0).rolling(14, min_periods=7).mean()
    down = -delta.clip(upper=0).rolling(14, min_periods=7).mean()
    rs = up / (down + 1e-9)
    df["RSI"] = 100 - (100 / (1 + rs))

    # Bollinger
    ma20 = df["Adj Close"].rolling(20, min_periods=10).mean()
    std20 = df["Adj Close"].rolling(20, min_periods=10).std()
    df["BB_middle"] = ma20
    df["BB_high"] = ma20 + 2*std20
    df["BB_low"] = ma20 - 2*std20

    # final dropna
    df = df.dropna().copy()
    return df

# ---------- FEATURE MATRIX ----------
def build_features_matrix(df, window=30):
    df = add_indicators(df, window)
    cols = ['Open','High','Low','Close','Adj Close','Volume',
            'MA3','MA5','Momentum','Volatility',
            'MACD','MACD_signal','RSI','BB_middle','BB_high','BB_low']
    if len(df) < window + 5:
        raise RuntimeError(f"Not enough data after indicators: have {len(df)}, need at least {window+5}")

    X_rows, y_class, y_reg, dates = [], [], [], []
    close = df['Adj Close'].values
    for i in range(window, len(df)-1):
        slice_ = df.iloc[i-window:i]
        flat = slice_[cols].values.flatten()
        X_rows.append(flat)
        y_class.append(1 if close[i+1] > close[i] else 0)
        y_reg.append(float(close[i+1]))
        dates.append(df.index[i])
    X = np.vstack(X_rows)
    return X, np.array(y_class), np.array(y_reg), dates

# ---------- MODEL TRAINING ----------
def train_and_save(ticker=TICKER, years=YEARS, window=WINDOW):
    print("Downloading data...")
    df = download_data(ticker, years)

    print("Building features...")
    X, y_cl, y_reg, dates = build_features_matrix(df, window)

    # time-series split (no shuffle)
    split = int(len(X) * 0.7)
    X_train, X_val = X[:split], X[split:]
    y_train_cl, y_val_cl = y_cl[:split], y_cl[split:]
    y_train_reg, y_val_reg = y_reg[:split], y_reg[split:]

    # scale
    scaler = StandardScaler().fit(X_train)
    Xtr = scaler.transform(X_train)
    Xv = scaler.transform(X_val)

    results = {}

    # SVM classifier
    print("Training SVM classifier...")
    svm = SVC(kernel='rbf', probability=True)
    svm.fit(Xtr, y_train_cl)
    svm_acc = float(accuracy_score(y_val_cl, svm.predict(Xv)))
    joblib.dump({'model': svm, 'scaler': scaler}, os.path.join(MODELS_DIR, f"{ticker}_svm.joblib"))
    results['svm_val_acc'] = svm_acc
    print("SVM val acc:", svm_acc)

    # SVR (regression)
    print("Training SVR...")
    svr = SVR(kernel='rbf')
    svr.fit(Xtr, y_train_reg)
    svr_mse = float(mean_squared_error(y_val_reg, svr.predict(Xv)))
    joblib.dump({'model': svr, 'scaler': scaler}, os.path.join(MODELS_DIR, f"{ticker}_svr.joblib"))
    results['svr_val_mse'] = svr_mse
    print("SVR val mse:", svr_mse)

    # MLP classifier
    print("Training MLP classifier...")
    mlp = MLPClassifier(hidden_layer_sizes=(128,64), max_iter=500)
    mlp.fit(Xtr, y_train_cl)
    mlp_acc = float(accuracy_score(y_val_cl, mlp.predict(Xv)))
    joblib.dump({'model': mlp, 'scaler': scaler}, os.path.join(MODELS_DIR, f"{ticker}_mlp.joblib"))
    results['mlp_val_acc'] = mlp_acc
    print("MLP val acc:", mlp_acc)

    # save scaler independently too
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"{ticker}_scaler.joblib"))

    # summary
    results['ticker'] = ticker
    results['trained_at'] = datetime.utcnow().isoformat() + "Z"
    results['n_samples'] = int(len(X))
    results['window'] = int(window)

    # save results
    with open(os.path.join(MODELS_DIR, f"{ticker}_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("Training finished. Results:", results)
    return results

# ---------- MAIN ----------
if __name__ == "__main__":
    res = train_and_save()
    # exit code 0 on success

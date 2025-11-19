import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import joblib
import datetime

TICKER = "AAPL"
WINDOW = 20

def make_features(df):
    df["Return"] = df["Close"].pct_change()
    df["MA10"] = df["Close"].rolling(10).mean()
    df["MA20"] = df["Close"].rolling(20).mean()
    df["Vol"] = df["Close"].rolling(20).std()

    # Label BEFORE dropping NaNs
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # Drop NaNs from all indicator rows
    df = df.dropna().copy()

    X = df[["Return","MA10","MA20","Vol"]].values
    y = df["Target"].values

    # Perfect alignment
    min_len = min(len(X), len(y))
    return X[:min_len], y[:min_len]

print("Downloading data...")
df = yf.download(TICKER, period="7y", auto_adjust=True, progress=False)

print("Building features...")
X, y = make_features(df)

print("Scaling...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Training...")
model = MLPClassifier(hidden_layer_sizes=(64,64), max_iter=1000)
model.fit(X_scaled, y)

print("Saving model...")
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("Training completed on", datetime.date.today())

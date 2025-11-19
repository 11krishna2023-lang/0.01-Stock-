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
    df = df.dropna()
    X = df[["Return","MA10","MA20","Vol"]].values
    y = (df["Close"].shift(-1) > df["Close"]).astype(int).dropna().values
    X = X[:-1]
    return X, y

print("Downloading data...")
df = yf.download(TICKER, period="7y")

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

# backtest.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import joblib
import json
from typing import Sequence, Tuple

def simple_vector_backtest(prices: Sequence[float], preds: Sequence[int], verbose=False) -> dict:
    """
    Simple vectorized backtest:
      - go long 1 unit when pred==1 at price t, exit next day at t+1
      - cash otherwise (no shorts)
    Returns metrics and time series.
    """
    prices = np.asarray(prices)
    preds = np.asarray(preds).astype(int)
    if len(preds) >= len(prices):
        preds = preds[:len(prices)-1]
    # returns aligned to preds length
    ret = np.zeros(len(preds))
    ret[preds == 1] = (prices[1:][preds == 1] - prices[:-1][preds == 1]) / prices[:-1][preds == 1]
    cumret = np.cumprod(1 + ret) - 1
    sr = np.mean(ret) / (np.std(ret) + 1e-9) * np.sqrt(252) if ret.std() > 0 else 0.0
    maxdd = compute_max_drawdown(cumret)
    stats = {
        "total_return": float(cumret[-1]) if len(cumret)>0 else 0.0,
        "sharpe_annual": float(sr),
        "max_drawdown": float(maxdd),
        "daily_returns_mean": float(np.mean(ret)),
        "daily_returns_std": float(np.std(ret))
    }
    if verbose:
        print("Backtest stats:", stats)
    return {"stats": stats, "cumret": cumret.tolist(), "daily_returns": ret.tolist()}

def compute_max_drawdown(cumret: np.ndarray) -> float:
    # cumret assumed cumulative returns (not prices)
    if len(cumret) == 0:
        return 0.0
    peak = np.maximum.accumulate(cumret + 1)
    dd = (cumret + 1) / peak - 1
    return float(np.min(dd))

def plot_backtest(cumret: Sequence[float], outpath: str = None):
    plt.figure(figsize=(8,4))
    plt.plot(cumret)
    plt.title("Cumulative return")
    plt.xlabel("Days")
    plt.ylabel("Cumulative return")
    plt.grid(True)
    if outpath:
        plt.tight_layout()
        plt.savefig(outpath)
    else:
        plt.show()


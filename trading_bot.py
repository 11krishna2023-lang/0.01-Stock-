# trading_bot.py
import os, time, joblib
from datetime import datetime, timedelta
import yfinance as yf
from notify import telegram_send
from your_predict_module import live_predict  # import from your V1/Colab code or copy function here
# or load models directly: joblib.load('models_output/AAPL_mlp.joblib')

ALPACA_KEY = os.getenv('PKLDMLK2CBKY24MP7SYBEUIDYM')
ALPACA_SECRET = os.getenv('DctPi76HXGraXYdxkA71XzugvXUsiJVo3ufMcL6jiPmp')
ALPACA_BASE = os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
USE_PAPER = os.getenv('USE_PAPER','true').lower() in ('1','true','yes')

# alpaca-py trading client
try:
    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import MarketOrderRequest
    from alpaca.trading.enums import OrderSide, TimeInForce
except Exception as e:
    raise RuntimeError("alpaca-py not installed or import failed: " + str(e))

# risk params
MAX_DAILY_TRADES = int(os.getenv('MAX_DAILY_TRADES', '5'))
POSITION_QTY = int(os.getenv('POSITION_QTY', '1'))
COOLDOWN_SECONDS = int(os.getenv('COOLDOWN_SECONDS', '60'))

def init_client():
    if not ALPACA_KEY or not ALPACA_SECRET:
        raise RuntimeError("Alpaca keys missing in env.")
    client = TradingClient(ALPACA_KEY, ALPACA_SECRET, paper=USE_PAPER)
    return client

def simple_trade_logic(client, ticker, model, scaler):
    # get latest data
    df = yf.download(ticker, period='60d', auto_adjust=True, progress=False)
    # compute features same as training pipeline then predict
    # For simplicity assume you have live_predict(model,scaler,ticker)
    res = live_predict(model, scaler, ticker=ticker)  # returns {'prediction':0/1, 'prob':0.x}
    pred = res['prediction']
    prob = res['prob']

    # basic thresholds
    if pred == 1 and prob > 0.6:
        side = OrderSide.BUY
    elif pred == 0 and prob < 0.4:
        side = OrderSide.SELL
    else:
        return {'status': 'no_trade', 'prob': prob}

    order = MarketOrderRequest(symbol=ticker, qty=POSITION_QTY, side=side, time_in_force=TimeInForce.DAY)
    try:
        resp = client.submit_order(order)
        telegram_send(f"Placed {side} order for {ticker} at {datetime.now()} prob={prob:.2f}")
        return {'status': 'order_submitted', 'order': str(resp), 'prob': prob}
    except Exception as e:
        telegram_send(f"Order failed: {e}")
        return {'status': 'error', 'error': str(e)}

def run_bot_loop(tickers, poll_seconds=60):
    client = init_client()
    # load models for each ticker (example load MLP)
    models = {}
    scalers = {}
    for t in tickers:
        # expect joblib files in models_output/
        models[t] = joblib.load(f"models_output/{t}_mlp.joblib")['model']
        scalers[t] = joblib.load(f"models_output/{t}_mlp.joblib")['scaler']
    daily_trades = 0
    last_trade_time = {}
    while True:
        for t in tickers:
            now = datetime.utcnow()
            if daily_trades >= MAX_DAILY_TRADES:
                print("Daily trade limit reached.")
                break
            # cooldown
            lt = last_trade_time.get(t)
            if lt and (now - lt).total_seconds() < COOLDOWN_SECONDS:
                continue
            try:
                res = simple_trade_logic(client, t, models[t], scalers[t])
                if res.get('status') == 'order_submitted':
                    daily_trades += 1
                    last_trade_time[t] = now
            except Exception as e:
                print("Error in trading loop for", t, e)
        time.sleep(poll_seconds)

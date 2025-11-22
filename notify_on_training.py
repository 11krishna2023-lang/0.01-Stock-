# notify_on_training.py
import os, json, glob
from notify import telegram_send, telegram_send_file

ARTIFACT_DIR = 'models_output'
summaries = glob.glob(os.path.join(ARTIFACT_DIR, '*_results.json'))
msgs = []
for s in summaries:
    d = json.load(open(s))
    msgs.append(f"{d.get('ticker')} accs: {d}")

# send summary
telegram_send("Daily training finished. Results:\n" + "\n".join(msgs))
# optionally send plots
for p in glob.glob(os.path.join(ARTIFACT_DIR, '*_backtest.png')):
    telegram_send_file(p, caption="Backtest")

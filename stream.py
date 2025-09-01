import websocket, json
from indicators import evaluate_signals
from config import SYMBOLS, TIMEFRAMES
from threading import Thread

signal_cache = {}

def start_stream(symbol, interval):
    url = f"wss://stream.binance.com:9443/ws/{symbol}@kline_{interval}"
    candles = []

    def on_message(ws, message):
        data = json.loads(message)
        k = data['k']
        if k['x']:
            candles.append({
                "timestamp": k['t'],
                "open":  float(k['o']),
                "high":  float(k['h']),
                "low":   float(k['l']),
                "close": float(k['c']),
                "volume":float(k['v']),
            })
            if len(candles) >= 100:
                signal = evaluate_signals(candles)
                signal_cache[(symbol, interval)] = signal
                candles.clear()

    ws = websocket.WebSocketApp(url, on_message=on_message)
    ws.run_forever()

def launch_streams():
    for symbol in SYMBOLS:
        for tf in TIMEFRAMES:
            Thread(target=start_stream, args=(symbol, tf)).start()

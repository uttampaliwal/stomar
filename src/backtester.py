import pandas as pd
import numpy as np
from src.portfolio import Portfolio


def run_backtest(df_feat, signals, initial_capital=100000, brokerage=0.0003):
    portfolio = Portfolio(initial_capital)
    in_position = {}
    tickers = list(signals.keys())

    for date_key in sorted(set(
        d for sigs in signals.values() for d in sigs.keys()
    )):
        date = pd.Timestamp(date_key)
        for ticker in tickers:
            if date_key not in signals[ticker]:
                continue
            signal = signals[ticker][date_key]
            price = signal.get("price", 0)
            if price == 0:
                continue

            if signal["direction"] == 1 and not in_position.get(ticker):
                qty = int(portfolio.cash * 0.25 / price)
                if qty > 0:
                    portfolio.buy(ticker, price, qty, date, brokerage)
                    in_position[ticker] = qty

            elif signal["direction"] == 0 and in_position.get(ticker):
                qty = in_position[ticker]
                portfolio.sell(ticker, price, qty, date, brokerage)
                in_position[ticker] = 0

    for ticker, qty in in_position.items():
        if qty and qty > 0:
            last_price = signals[ticker].get(
                max(signals[ticker].keys()), {}
            ).get("price", 0)
            if last_price > 0:
                portfolio.sell(ticker, last_price, qty, pd.Timestamp.now(), brokerage)

    return portfolio.get_stats(), portfolio


def generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feature_cols):
    import torch
    from src.model import DEVICE

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols].dropna()
    scaled = scaler.transform(data.values)

    signals = {}
    for i in range(60, len(scaled)):
        date = str(data.index[i].date())
        inp = torch.tensor(scaled[i-60:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        prev_close = scaled[i-1, 0]
        actual_close = scaled[i, 0]

        lstm.eval()
        gru.eval()
        transformer.eval()
        with torch.no_grad():
            p_l = lstm(inp).item()
            p_g = gru(inp).item()
            p_t = transformer(inp).item()

        d_l = 1 if p_l > prev_close else 0
        d_g = 1 if p_g > prev_close else 0
        d_t = 1 if p_t > prev_close else 0

        xgb_inp = data.iloc[[i-1]][feature_cols]
        xgb_p = xgb.predict_proba(xgb_inp)[0][1]

        ens = d_l + d_g + d_t
        dl_dir = 1 if ens >= 2 else 0
        final = 1 if (dl_dir + xgb_p) / 2 > 0.5 else 0
        conf = abs((dl_dir + xgb_p) / 2 - 0.5) * 2

        signals[date] = {
            "direction": final,
            "confidence": conf,
            "price": float(data.loc[data.index[i], "close"]),
            "actual": 1 if actual_close > prev_close else 0,
        }
    return {ticker: signals}

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "WIPRO.NS", "AXISBANK.NS", "TITAN.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "NTPC.NS", "ONGC.NS",
]

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        last_date = df.index[-1]
        if last_date >= pd.Timestamp.now().normalize() - pd.Timedelta(days=2):
            return df

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    df.to_parquet(cache_path)
    return df


def get_live_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    return float(data["Close"].iloc[-1])


def get_market_status() -> str:
    now = datetime.now()
    if now.weekday() >= 5:
        return "Closed (Weekend)"
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if market_open <= now <= market_close:
        return "Open"
    return "Closed"

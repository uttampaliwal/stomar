from __future__ import annotations

from typing import Any

import pandas as pd
import yfinance as yf


def fetch_free_historical_data(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch free historical OHLCV data using Yahoo Finance."""
    data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if data is None or data.empty:
        raise ValueError(f"Could not fetch free historical data for {ticker}")
    data = data.reset_index()
    data.columns = [c.lower() for c in data.columns]
    if "date" not in data.columns and "datetime" in data.columns:
        data = data.rename(columns={"datetime": "date"})
    if "close" not in data.columns:
        raise ValueError(f"No close column found for {ticker}")
    return data


def get_free_latest_quote(ticker: str) -> dict[str, Any]:
    data = fetch_free_historical_data(ticker, period="5d", interval="1d")
    if data.empty:
        return {"price": None}
    latest = data.iloc[-1]
    return {
        "price": float(latest["close"]),
        "date": str(latest.get("date", "")),
    }

import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import time
import logging

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)

_fetch_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_FETCH_CACHE_TTL = 600  # 10 minutes in-memory cache


def _cache_get(key: str) -> pd.DataFrame | None:
    if key in _fetch_cache:
        ts, df = _fetch_cache[key]
        if time.time() - ts < _FETCH_CACHE_TTL:
            return df
        del _fetch_cache[key]
    return None


def _cache_put(key: str, df: pd.DataFrame):
    _fetch_cache[key] = (time.time(), df)

NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "WIPRO.NS", "AXISBANK.NS", "TITAN.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "NTPC.NS", "ONGC.NS",
]

os.makedirs(DATA_DIR, exist_ok=True)


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = f"{ticker}:{period}:{interval}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        last_date = df.index[-1]
        if last_date.tz is not None:
            now = pd.Timestamp.now(tz=last_date.tz)
        else:
            now = pd.Timestamp.now()
        # Refresh if data is stale (>2 days old) OR too short (< expected rows for period)
        expected_min_rows = {"1y": 200, "2y": 400, "3y": 600, "5y": 1000}
        min_rows = expected_min_rows.get(period, 200)
        if last_date >= now.normalize() - pd.Timedelta(days=2) and len(df) >= min_rows:
            _cache_put(cache_key, df)
            return df

    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)

    if df.empty:
        raise ValueError(f"No data found for ticker: {ticker}")

    df.columns = [c.lower() for c in df.columns]
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"

    df.to_parquet(cache_path)
    _cache_put(cache_key, df)
    return df


def get_live_price(ticker: str) -> float:
    stock = yf.Ticker(ticker)
    data = stock.history(period="1d", interval="1m")
    if data.empty or "Close" not in data.columns:
        logger.warning("No live price data for %s", ticker)
        return 0.0
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


def fetch_with_validation(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict:
    """Fetch stock data with validation.

    Returns:
        Dict with df, validation, ticker
    """
    from src.data.data_validation import validate_data

    df = fetch_stock_data(ticker, period=period, interval=interval, force_refresh=force_refresh)
    validation = validate_data(df, ticker)

    return {
        "df": df,
        "validation": validation,
        "ticker": ticker,
    }

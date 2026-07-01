"""Portfolio optimizer endpoint."""

from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.trading.optimizer import optimize_portfolio
from api.utils import parallel_fetch

router = APIRouter()


def _fetch_close(ticker):
    df = fetch_stock_data(ticker, period="1y")
    if df is not None and not df.empty:
        return df["close"]
    return None


def _sanitize(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    return obj


@router.get("/")
def optimize():
    try:
        raw = parallel_fetch(_fetch_close, NSE_STOCKS, max_workers=8)
        prices = {k: v for k, v in raw.items() if v is not None}

        if len(prices) < 3:
            return {"error": "Not enough stocks with data"}

        import pandas as pd
        prices_df = pd.DataFrame(prices)
        result = optimize_portfolio(prices_df)

        if result is None:
            return {"error": "Optimization failed"}

        return _sanitize(result)
    except Exception as e:
        return {"error": str(e)}

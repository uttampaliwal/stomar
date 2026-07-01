"""Correlation matrix endpoint."""

from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from api.utils import parallel_fetch

router = APIRouter()


def _fetch_close(ticker):
    df = fetch_stock_data(ticker, period="1y")
    if df is not None and len(df) > 30:
        return df["close"]
    return None


@router.get("/")
def correlation_matrix():
    try:
        import pandas as pd
        raw = parallel_fetch(_fetch_close, NSE_STOCKS, max_workers=8)
        prices = {k: v for k, v in raw.items() if v is not None}

        if len(prices) < 2:
            return {"error": "Not enough stocks with data"}

        prices_df = pd.DataFrame(prices)
        # Drop columns with all NaN, then drop rows with any NaN
        prices_df = prices_df.dropna(axis=1, how="all").dropna()
        if len(prices_df) < 30:
            return {"error": "Insufficient overlapping data"}
        returns = prices_df.pct_change().dropna()
        corr = returns.corr()

        tickers = corr.columns.tolist()
        matrix = corr.values.tolist()

        return {
            "tickers": tickers,
            "matrix": matrix,
        }
    except Exception as e:
        return {"error": str(e)}

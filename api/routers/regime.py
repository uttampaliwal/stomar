"""Regime detection and strategy endpoint."""

from fastapi import APIRouter, Query
from src.data_fetcher import fetch_stock_data
from src.regime import detect_regime

router = APIRouter()


@router.get("/{ticker}")
def get_regime(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        regime = detect_regime(df["Close"])
        return {"ticker": ticker, **regime}
    except Exception as e:
        return {"error": str(e)}

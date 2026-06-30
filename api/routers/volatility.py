"""Volatility analysis endpoint."""

from fastapi import APIRouter, Query
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.volatility import full_volatility_analysis

router = APIRouter()


@router.get("/{ticker}")
def volatility_analysis(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        result = full_volatility_analysis(df)
        if result is None:
            return {"error": "Volatility analysis failed"}

        return {"ticker": ticker, **result}
    except Exception as e:
        return {"error": str(e)}

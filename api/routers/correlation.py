"""Correlation matrix endpoint."""

import numpy as np
from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS

router = APIRouter()


@router.get("/")
def correlation_matrix():
    try:
        import pandas as pd
        prices = {}
        for ticker in NSE_STOCKS:
            try:
                df = fetch_stock_data(ticker, period="1y")
                if df is not None and len(df) > 30:
                    prices[ticker] = df["close"]
            except Exception:
                continue

        if len(prices) < 2:
            return {"error": "Not enough stocks with data"}

        prices_df = pd.DataFrame(prices)
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

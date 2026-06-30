"""Portfolio optimizer endpoint."""

import numpy as np
from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.optimizer import optimize_portfolio

router = APIRouter()


@router.get("/")
def optimize():
    try:
        prices = {}
        for ticker in NSE_STOCKS:
            try:
                df = fetch_stock_data(ticker, period="1y")
                if df is not None and not df.empty:
                    prices[ticker] = df["close"]
            except Exception:
                continue

        if len(prices) < 3:
            return {"error": "Not enough stocks with data"}

        import pandas as pd
        prices_df = pd.DataFrame(prices)
        result = optimize_portfolio(prices_df)

        if result is None:
            return {"error": "Optimization failed"}

        return result
    except Exception as e:
        return {"error": str(e)}

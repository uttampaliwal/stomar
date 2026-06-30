"""Backtest endpoint."""

from fastapi import APIRouter, Query
from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.trainer import FEATURE_COLS
from src.backtester import run_walk_forward_backtest

router = APIRouter()


@router.get("/{ticker}")
def run_backtest(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}
        df_feat = add_technical_indicators(df.copy(), ticker)
        result = run_walk_forward_backtest(df_feat, ticker, FEATURE_COLS)
        if result is None:
            return {"error": "Backtest failed"}
        return result
    except Exception as e:
        return {"error": str(e)}

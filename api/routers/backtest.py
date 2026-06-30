"""Backtest endpoint."""

from fastapi import APIRouter
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
        result = run_walk_forward_backtest(ticker, df_feat, FEATURE_COLS)

        if result is None:
            return {"error": "Backtest failed — not enough data"}

        stats, portfolio = result
        if not stats or stats.get("total_trades", 0) == 0:
            return {"error": "Backtest produced no trades"}

        return {
            "ticker": ticker,
            "ensemble_accuracy": stats.get("ensemble_accuracy", 0),
            "annual_return": stats.get("annual_return", 0),
            "sharpe": stats.get("sharpe", 0),
            "sortino": stats.get("sortino", 0),
            "max_drawdown": stats.get("max_drawdown", 0),
            "total_return": stats.get("total_return", 0),
            "total_trades": stats.get("total_trades", 0),
            "win_rate": stats.get("win_rate", 0),
        }
    except Exception as e:
        return {"error": str(e)}

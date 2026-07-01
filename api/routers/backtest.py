"""Backtest endpoint."""

from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data
from src.data.features import add_technical_indicators
from src.models.trainer import FEATURE_COLS
from src.trading.backtester import run_walk_forward_backtest

router = APIRouter()


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


@router.get("/{ticker}")
def run_backtest(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}
        df_feat = add_technical_indicators(df.copy(), ticker)
        stats, portfolio, test_results = run_walk_forward_backtest(ticker, df_feat, FEATURE_COLS)

        if stats is None:
            return {"error": "Backtest failed — not enough data"}

        if not stats or stats.get("total_trades", 0) == 0:
            return {"error": "Backtest produced no trades"}

        response = {
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

        if test_results:
            response["test_results"] = [
                _sanitize({
                    "window": t.get("window", f"W{i+1}") if isinstance(t, dict) else f"W{i+1}",
                    "ensemble_accuracy": t.get("ensemble_accuracy", 0),
                    "total_return": t.get("total_return", 0),
                    "sharpe": t.get("sharpe", 0),
                    "total_trades": t.get("total_trades", 0),
                })
                for i, t in enumerate(test_results)
            ]

        if portfolio and isinstance(portfolio, dict):
            mc = portfolio.get("monte_carlo")
            if mc:
                response["monte_carlo"] = _sanitize(mc)

        return _sanitize(response)
    except Exception as e:
        return {"error": str(e)}

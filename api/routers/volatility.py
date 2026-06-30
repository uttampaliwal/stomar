"""Volatility analysis endpoint."""

import numpy as np
import pandas as pd
from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data
from src.volatility import full_volatility_analysis

router = APIRouter()


def _to_native(obj):
    """Recursively convert numpy/pandas types to JSON-safe Python primitives."""
    if isinstance(obj, dict):
        return {str(k): _to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_native(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return round(float(obj), 6)
    if isinstance(obj, np.ndarray):
        return [_to_native(v) for v in obj.tolist()]
    if isinstance(obj, pd.Timestamp):
        return str(obj)
    if isinstance(obj, pd.Series):
        return [_to_native(v) for v in obj.values]
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if hasattr(obj, 'item'):
        return _to_native(obj.item())
    if isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    return str(obj)


@router.get("/{ticker}")
def volatility_analysis(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        result = full_volatility_analysis(df)
        if result is None:
            return {"error": "Volatility analysis failed"}

        return {"ticker": ticker, **_to_native(result)}
    except Exception as e:
        return {"error": str(e)}

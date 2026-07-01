"""Scenario analysis endpoint."""

from fastapi import APIRouter, Query
from src.data.data_fetcher import fetch_stock_data
from src.signals.scenarios import run_all_scenarios

router = APIRouter()


@router.get("/{ticker}")
def run_scenarios(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        result = run_all_scenarios(df)
        if result is None:
            return {"error": "Scenario analysis failed"}

        return {"ticker": ticker, "scenarios": result}
    except Exception as e:
        return {"error": str(e)}

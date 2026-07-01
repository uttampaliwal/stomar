"""Holdings and MF tracker endpoint."""

import os
from fastapi import APIRouter, UploadFile, File
from src.trading.holdings import parse_holdings_csv, compute_portfolio_stats

router = APIRouter()


@router.get("/stats")
def holdings_stats():
    try:
        holdings_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "holdings.csv")
        if not os.path.exists(holdings_path):
            return {"error": "No holdings file found"}
        df = parse_holdings_csv(holdings_path)
        stats = compute_portfolio_stats(df)
        return {"stats": stats, "holdings": df.to_dict(orient="records")}
    except Exception as e:
        return {"error": str(e)}

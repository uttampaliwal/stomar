"""Pipeline management endpoint."""

from fastapi import APIRouter
from src.model import models_exist
from src.data_fetcher import NSE_STOCKS

router = APIRouter()


@router.get("/status")
def pipeline_status():
    try:
        trained = [t for t in NSE_STOCKS if models_exist(t)]
        missing = [t for t in NSE_STOCKS if not models_exist(t)]
        return {
            "total": len(NSE_STOCKS),
            "trained": len(trained),
            "missing": len(missing),
            "trained_tickers": trained,
            "missing_tickers": missing,
        }
    except Exception as e:
        return {"error": str(e)}

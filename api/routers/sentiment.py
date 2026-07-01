"""Sentiment analysis endpoint."""

from fastapi import APIRouter, Query
from src.signals.sentiment import get_stock_sentiment

router = APIRouter()


@router.get("/{ticker}")
def stock_sentiment(ticker: str):
    try:
        result = get_stock_sentiment(ticker)
        if isinstance(result, dict):
            return {
                "ticker": ticker,
                "score": round(float(result.get("weighted_score", 0)), 4),
                "label": result.get("label", "Neutral"),
                "headlines": result.get("headlines", []),
                "source_scores": result.get("source_scores", {}),
            }
        return {"ticker": ticker, "score": 0, "label": "Neutral"}
    except Exception as e:
        return {"error": str(e)}

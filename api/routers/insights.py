from fastapi import APIRouter

from src.analytics.market_insights import build_recommendation_from_history
from src.data.data_fetcher import fetch_stock_data, get_live_price
from src.signals.sentiment import get_stock_sentiment

router = APIRouter()


@router.get("/recommend/{ticker}")
def recommend(ticker: str):
    try:
        frame = fetch_stock_data(ticker, period="2y", interval="1d", force_refresh=False)
        if frame is None or frame.empty:
            return {"error": f"No data for {ticker}"}
        result = build_recommendation_from_history(frame, ticker=ticker)
        return result
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}


@router.get("/enrichment/{ticker}")
def enrichment(ticker: str):
    try:
        frame = fetch_stock_data(ticker, period="2mo", interval="1d", force_refresh=False)
        latest_price = None
        change_pct = 0.0
        if frame is not None and not frame.empty:
            close = frame["close"].astype(float)
            latest_price = float(close.iloc[-1])
            if len(close) >= 2:
                prev_price = float(close.iloc[-2])
                change_pct = ((latest_price - prev_price) / prev_price * 100.0) if prev_price else 0.0

        live_price = get_live_price(ticker)
        if live_price and live_price > 0:
            latest_price = live_price

        sentiment = get_stock_sentiment(ticker)
        return {
            "ticker": ticker,
            "price": round(latest_price, 2) if latest_price is not None else None,
            "change_pct": round(change_pct, 2),
            "sentiment_score": round(float(sentiment.get("score", 0) or 0), 3),
            "sentiment_label": sentiment.get("label", "Neutral"),
            "headline_count": int(sentiment.get("total", 0) or 0),
            "headlines": (sentiment.get("articles") or [])[:3],
        }
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}

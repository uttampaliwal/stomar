"""Consensus signal endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS
from src.regime import detect_regime
from src.sentiment import get_stock_sentiment
from api.utils import parallel_fetch

router = APIRouter()


def _load(ticker):
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


def _consensus_one(ticker):
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        return None
    df_feat = add_technical_indicators(df.copy(), ticker)

    ensemble_signal = "HOLD"
    ensemble_conf = 0.0
    if models_exist(ticker):
        m = _load(ticker)
        direction, confidence, details = predict_ensemble(
            m["lstm"], m["gru"], m["transformer"],
            m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
            lgb_model=m["lgb"],
        )
        ensemble_signal = "BUY" if direction == 1 else "SELL"
        ensemble_conf = round(float(details.get("ensemble_prob", confidence / 100)) if details else float(confidence) / 100, 4)

    regime = detect_regime(df["close"])
    regime_name = regime.get("regime", "Unknown")

    sentiment = get_stock_sentiment(ticker)
    sentiment_score = sentiment.get("weighted_score", 0) if isinstance(sentiment, dict) else 0

    consensus = "HOLD"
    if ensemble_signal == "BUY" and regime_name == "Bull" and sentiment_score > 0:
        consensus = "STRONG BUY"
    elif ensemble_signal == "BUY" and sentiment_score > 0:
        consensus = "BUY"
    elif ensemble_signal == "SELL" and regime_name == "Bear":
        consensus = "STRONG SELL"
    elif ensemble_signal == "SELL":
        consensus = "SELL"
    else:
        consensus = "CONFLICTED"

    return {
        "ticker": ticker,
        "ensemble_signal": ensemble_signal,
        "ensemble_confidence": ensemble_conf,
        "regime": regime_name,
        "sentiment": round(float(sentiment_score), 4),
        "consensus": consensus,
    }


@router.get("/")
def get_consensus():
    try:
        raw = parallel_fetch(_consensus_one, NSE_STOCKS, max_workers=8)
        results = [v for v in raw.values() if v is not None]

        buy_count = sum(1 for r in results if "BUY" in r["consensus"])
        sell_count = sum(1 for r in results if "SELL" in r["consensus"])
        hold_count = sum(1 for r in results if r["consensus"] == "HOLD")
        conflicted = sum(1 for r in results if r["consensus"] == "CONFLICTED")

        return {
            "results": results,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "hold_count": hold_count,
            "conflicted": conflicted,
        }
    except Exception as e:
        return {"error": str(e)}

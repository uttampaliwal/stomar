"""Consensus signal endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS
from src.regime import detect_regime
from src.sentiment import get_stock_sentiment
from src.flow import get_flow_sentiment, fetch_options_pcr
from src.multitimeframe import fetch_mtf_data, get_combined_signal

router = APIRouter()


@router.get("/")
def get_consensus():
    try:
        results = []
        for ticker in NSE_STOCKS:
            try:
                df = fetch_stock_data(ticker)
                if df is None or df.empty:
                    continue
                df_feat = add_technical_indicators(df.copy(), ticker)

                ensemble_signal = "HOLD"
                ensemble_conf = 0.0
                if models_exist(ticker):
                    models = load_models(ticker)
                    if models:
                        direction, confidence, _ = predict_ensemble(
                            models["lstm"], models["gru"], models["transformer"],
                            models["xgb"], models["scaler"], FEATURE_COLS, df_feat,
                            lgb_model=models.get("lgb"),
                        )
                        ensemble_signal = direction
                        ensemble_conf = round(float(confidence), 4)

                regime = detect_regime(df["Close"])
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

                results.append({
                    "ticker": ticker,
                    "ensemble_signal": ensemble_signal,
                    "ensemble_confidence": ensemble_conf,
                    "regime": regime_name,
                    "sentiment": round(float(sentiment_score), 4),
                    "consensus": consensus,
                })
            except Exception:
                continue

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

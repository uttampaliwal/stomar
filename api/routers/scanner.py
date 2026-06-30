"""Multi-stock scanner endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS

router = APIRouter()


@router.get("/")
def scan_stocks():
    try:
        results = []
        for ticker in NSE_STOCKS:
            try:
                df = fetch_stock_data(ticker)
                if df is None or df.empty:
                    continue
                df_feat = add_technical_indicators(df.copy(), ticker)
                last = df_feat.iloc[-1]

                if not models_exist(ticker):
                    continue
                models = load_models(ticker)
                if not models:
                    continue

                direction, confidence, _ = predict_ensemble(
                    models["lstm"], models["gru"], models["transformer"],
                    models["xgb"], models["scaler"], FEATURE_COLS, df_feat,
                    lgb_model=models.get("lgb"),
                )

                day_return = round(float((last["Close"] - last["Open"]) / last["Open"] * 100), 2) if last["Open"] > 0 else 0
                results.append({
                    "ticker": ticker,
                    "signal": direction,
                    "confidence": round(float(confidence), 4),
                    "price": round(float(last["Close"]), 2),
                    "day_return": day_return,
                    "rsi": round(float(last.get("RSI", 50)), 2),
                })
            except Exception:
                continue

        buy_count = sum(1 for r in results if r["signal"] == "BUY")
        sell_count = sum(1 for r in results if r["signal"] == "SELL")

        return {
            "results": results,
            "buy_count": buy_count,
            "sell_count": sell_count,
        }
    except Exception as e:
        return {"error": str(e)}

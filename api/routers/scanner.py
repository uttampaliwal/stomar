"""Multi-stock scanner endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS
from api.utils import parallel_fetch

router = APIRouter()


def _load(ticker):
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


def _scan_one(ticker):
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        return None
    df_feat = add_technical_indicators(df.copy(), ticker)
    last = df_feat.iloc[-1]

    if not models_exist(ticker):
        return None
    m = _load(ticker)

    direction, confidence, _ = predict_ensemble(
        m["lstm"], m["gru"], m["transformer"],
        m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
        lgb_model=m["lgb"],
    )
    direction_label = "BUY" if direction == 1 else "SELL"

    day_return = round(float((last["close"] - last["open"]) / last["open"] * 100), 2) if last["open"] > 0 else 0
    return {
        "ticker": ticker,
        "signal": direction_label,
        "confidence": round(float(confidence), 4),
        "price": round(float(last["close"]), 2),
        "day_return": day_return,
        "rsi": round(float(last.get("rsi", 50)), 2),
    }


@router.get("/")
def scan_stocks():
    try:
        raw = parallel_fetch(_scan_one, NSE_STOCKS, max_workers=8)
        results = [v for v in raw.values() if v is not None]

        buy_count = sum(1 for r in results if r["signal"] == "BUY")
        sell_count = sum(1 for r in results if r["signal"] == "SELL")

        return {
            "results": results,
            "buy_count": buy_count,
            "sell_count": sell_count,
        }
    except Exception as e:
        return {"error": str(e)}

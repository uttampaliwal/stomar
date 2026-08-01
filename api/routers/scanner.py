"""Multi-stock scanner endpoint."""

from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.data.features import add_technical_indicators
from src.models.model import load_models, models_exist, model_feature_cols
from src.models.ensemble import predict_ensemble
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
    feature_cols = model_feature_cols(m)  # model's own trained schema

    direction, confidence, details = predict_ensemble(
        m["lstm"], m["gru"], m["transformer"],
        m["xgb"], m["scaler"], feature_cols, df_feat,
        lgb_model=m["lgb"],
    )
    direction_label = "BUY" if direction == 1 else "SELL"
    ensemble_prob = float(details.get("ensemble_prob", confidence / 100)) if details else float(confidence) / 100

    day_return = round(float((last["close"] - last["open"]) / last["open"] * 100), 2) if last["open"] > 0 else 0
    chg_5d = 0.0
    if len(df_feat) >= 6:
        close_now = float(last["close"])
        close_5d = float(df_feat.iloc[-6]["close"])
        if close_5d > 0:
            chg_5d = round((close_now / close_5d - 1) * 100, 2)

    return {
        "ticker": ticker,
        "signal": direction_label,
        "confidence": round(ensemble_prob, 4),
        "price": round(float(last["close"]), 2),
        "day_return": day_return,
        "chg_5d": chg_5d,
        "rsi": round(float(last.get("rsi", 50)), 2),
    }


@router.get("/")
def scan_stocks():
    try:
        trained_count = sum(1 for t in NSE_STOCKS if models_exist(t))
        raw = parallel_fetch(_scan_one, NSE_STOCKS, max_workers=8)
        results = [v for v in raw.values() if v is not None]

        buy_count = sum(1 for r in results if r["signal"] == "BUY")
        sell_count = sum(1 for r in results if r["signal"] == "SELL")

        return {
            "results": results,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "scanned": len(results),
            "total": len(NSE_STOCKS),
            "trained": trained_count,
        }
    except Exception as e:
        return {"error": str(e)}

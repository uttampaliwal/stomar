"""Stock predictions endpoint."""

import os
import numpy as np
from fastapi import APIRouter, Query
from src.data.data_fetcher import fetch_stock_data
from src.data.features import add_technical_indicators
from src.models.model import load_models, models_exist
from src.models.trainer import FEATURE_COLS
from src.models.ensemble import predict_ensemble

router = APIRouter()


def _load(ticker):
    """Load models and return as a dict."""
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


@router.get("/{ticker}")
def get_prediction(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        df_feat = add_technical_indicators(df.copy(), ticker)
        last = df_feat.iloc[-1]

        metrics = {
            "price": round(float(last["close"]), 2),
            "day_change": round(float(last["close"] - last["open"]), 2),
            "day_change_pct": round(float((last["close"] - last["open"]) / last["open"] * 100), 2),
            "volume": int(last["volume"]),
            "atr": round(float(last.get("atr", 0)), 2),
            "rsi": round(float(last.get("rsi", 50)), 2),
            "macd": round(float(last.get("macd", 0)), 4),
        }

        prediction = None
        if models_exist(ticker):
            m = _load(ticker)
            direction, confidence, details = predict_ensemble(
                m["lstm"], m["gru"], m["transformer"],
                m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
                lgb_model=m["lgb"],
            )
            direction_label = "BUY" if direction == 1 else "SELL"
            prediction = {
                "direction": direction_label,
                "confidence": round(float(details.get("ensemble_prob", confidence / 100)), 4),
                "conviction": round(float(confidence), 4),
                "details": {k: (round(float(v), 4) if isinstance(v, (int, float)) else str(v)) for k, v in details.items()} if details else {},
            }

        recent = df_feat.tail(7)[["open", "high", "low", "close", "volume"]].reset_index(drop=False)
        recent_data = []
        for _, row in recent.iterrows():
            recent_data.append({
                "date": str(row.get("date", row.index[0])),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            })

        ohlcv = df_feat[["open", "high", "low", "close", "volume"]].tail(120).reset_index(drop=False)
        chart_data = []
        for _, row in ohlcv.iterrows():
            chart_data.append({
                "date": str(row.get("date", row.index[0])),
                "open": round(float(row["open"]), 2),
                "high": round(float(row["high"]), 2),
                "low": round(float(row["low"]), 2),
                "close": round(float(row["close"]), 2),
                "volume": int(row["volume"]),
            })

        return {
            "ticker": ticker,
            "metrics": metrics,
            "prediction": prediction,
            "recent": recent_data,
            "chart": chart_data,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/{ticker}/feature-importance")
def get_feature_importance(ticker: str):
    try:
        if not models_exist(ticker):
            return {"error": "Models not trained"}
        m = _load(ticker)
        xgb = m["xgb"]
        importances = xgb.feature_importances_
        features = FEATURE_COLS[:len(importances)]
        pairs = sorted(zip(features, importances.tolist()), key=lambda x: -x[1])[:10]
        return {"features": [{"name": n, "importance": round(v, 4)} for n, v in pairs]}
    except Exception as e:
        return {"error": str(e)}

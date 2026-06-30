"""Stock predictions endpoint."""

import os
import numpy as np
from fastapi import APIRouter, Query
from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.trainer import FEATURE_COLS
from src.ensemble import predict_ensemble

router = APIRouter()


@router.get("/{ticker}")
def get_prediction(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        df_feat = add_technical_indicators(df.copy(), ticker)
        last = df_feat.iloc[-1]

        metrics = {
            "price": round(float(last["Close"]), 2),
            "day_change": round(float(last["Close"] - last["Open"]), 2),
            "day_change_pct": round(float((last["Close"] - last["Open"]) / last["Open"] * 100), 2),
            "volume": int(last["Volume"]),
            "atr": round(float(last.get("ATR", 0)), 2),
            "rsi": round(float(last.get("RSI", 50)), 2),
            "macd": round(float(last.get("MACD", 0)), 4),
        }

        prediction = None
        if models_exist(ticker):
            models = load_models(ticker)
            if models:
                direction, confidence, details = predict_ensemble(
                    models["lstm"], models["gru"], models["transformer"],
                    models["xgb"], models["scaler"], FEATURE_COLS, df_feat,
                    lgb_model=models.get("lgb"),
                )
                prediction = {
                    "direction": direction,
                    "confidence": round(float(confidence), 4),
                    "details": {k: round(float(v), 4) for k, v in details.items()} if details else {},
                }

        recent = df_feat.tail(7)[["Open", "High", "Low", "Close", "Volume"]].reset_index(drop=False)
        recent_data = []
        for _, row in recent.iterrows():
            recent_data.append({
                "date": str(row.get("Date", row.index[0])),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
            })

        ohlcv = df_feat[["Open", "High", "Low", "Close", "Volume"]].tail(120).reset_index(drop=False)
        chart_data = []
        for _, row in ohlcv.iterrows():
            chart_data.append({
                "date": str(row.get("Date", row.index[0])),
                "open": round(float(row["Open"]), 2),
                "high": round(float(row["High"]), 2),
                "low": round(float(row["Low"]), 2),
                "close": round(float(row["Close"]), 2),
                "volume": int(row["Volume"]),
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
        models = load_models(ticker)
        if not models or "xgb" not in models:
            return {"error": "XGBoost model not available"}
        xgb = models["xgb"]
        importances = xgb.feature_importances_
        features = FEATURE_COLS[:len(importances)]
        pairs = sorted(zip(features, importances.tolist()), key=lambda x: -x[1])[:10]
        return {"features": [{"name": n, "importance": round(v, 4)} for n, v in pairs]}
    except Exception as e:
        return {"error": str(e)}

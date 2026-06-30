"""Stock ranking endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS
from src.ranking import rank_stocks

router = APIRouter()


@router.get("/")
def get_rankings():
    try:
        stock_data = {}
        ml_signals = {}
        for ticker in NSE_STOCKS:
            try:
                df = fetch_stock_data(ticker)
                if df is None or df.empty:
                    continue
                df_feat = add_technical_indicators(df.copy(), ticker)
                stock_data[ticker] = df_feat

                if models_exist(ticker):
                    models = load_models(ticker)
                    if models:
                        direction, confidence, _ = predict_ensemble(
                            models["lstm"], models["gru"], models["transformer"],
                            models["xgb"], models["scaler"], FEATURE_COLS, df_feat,
                            lgb_model=models.get("lgb"),
                        )
                        ml_signals[ticker] = {"direction": direction, "confidence": float(confidence)}
            except Exception:
                continue

        if not stock_data:
            return {"error": "No stock data available"}

        rankings = rank_stocks(stock_data, ml_signals=ml_signals)
        return {"rankings": rankings}
    except Exception as e:
        return {"error": str(e)}

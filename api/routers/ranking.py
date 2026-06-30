"""Stock ranking endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.ensemble import predict_ensemble
from src.trainer import FEATURE_COLS
from src.ranking import rank_stocks

router = APIRouter()


def _load(ticker):
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


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
                    m = _load(ticker)
                    direction, confidence, _ = predict_ensemble(
                        m["lstm"], m["gru"], m["transformer"],
                        m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
                        lgb_model=m["lgb"],
                    )
                    ml_signals[ticker] = 1.0 if direction == 1 else -1.0
            except Exception:
                continue

        if not stock_data:
            return {"error": "No stock data available"}

        rankings = rank_stocks(stock_data, ml_signals=ml_signals)
        return {"rankings": rankings}
    except Exception as e:
        return {"error": str(e)}

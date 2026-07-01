"""Stock ranking endpoint."""

from fastapi import APIRouter
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.data.features import add_technical_indicators
from src.models.model import load_models, models_exist
from src.models.ensemble import predict_ensemble
from src.models.trainer import FEATURE_COLS
from src.signals.ranking import rank_stocks
from api.utils import parallel_fetch

router = APIRouter()


def _load(ticker):
    result = load_models(ticker)
    lstm, gru, transformer, xgb, scaler, features, lgb = result
    return {"lstm": lstm, "gru": gru, "transformer": transformer, "xgb": xgb, "scaler": scaler, "features": features, "lgb": lgb}


def _fetch_and_predict(ticker):
    df = fetch_stock_data(ticker)
    if df is None or df.empty:
        return None
    df_feat = add_technical_indicators(df.copy(), ticker)

    ml_signal = 0.0
    if models_exist(ticker):
        m = _load(ticker)
        direction, confidence, _ = predict_ensemble(
            m["lstm"], m["gru"], m["transformer"],
            m["xgb"], m["scaler"], FEATURE_COLS, df_feat,
            lgb_model=m["lgb"],
        )
        ml_signal = 1.0 if direction == 1 else -1.0
    return {"df": df_feat, "ml_signal": ml_signal}


@router.get("/")
def get_rankings():
    try:
        raw = parallel_fetch(_fetch_and_predict, NSE_STOCKS, max_workers=8)

        stock_data = {}
        ml_signals = {}
        for ticker, result in raw.items():
            if result is not None:
                stock_data[ticker] = result["df"]
                ml_signals[ticker] = result["ml_signal"]

        if not stock_data:
            return {"error": "No stock data available"}

        rankings = rank_stocks(stock_data, ml_signals=ml_signals)
        return {"rankings": rankings}
    except Exception as e:
        return {"error": str(e)}

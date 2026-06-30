"""Volatility analysis endpoint."""

from fastapi import APIRouter
from src.data_fetcher import fetch_stock_data
from src.volatility import (
    historical_volatility, ewma_volatility, parkinson_volatility,
    garman_klass_volatility, yang_zhang_volatility,
    detect_volatility_regime, forecast_volatility,
    compute_bollinger_bands, compute_atr,
)

router = APIRouter()


def _safe_float(val, default=0.0):
    try:
        if hasattr(val, 'iloc'):
            return float(val.iloc[-1])
        if hasattr(val, 'item'):
            return float(val.item())
        return float(val)
    except Exception:
        return default


@router.get("/{ticker}")
def volatility_analysis(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        close = df["close"]
        high = df["high"]
        low = df["low"]
        open_ = df["open"]
        returns = close.pct_change().dropna()

        has_data = len(returns) > 20
        hist_vol = _safe_float(historical_volatility(returns).iloc[-1]) if has_data else 0
        ewma_vol = _safe_float(ewma_volatility(returns).iloc[-1]) if has_data else 0
        park_vol = _safe_float(parkinson_volatility(high, low).iloc[-1]) if has_data else 0
        gk_vol = _safe_float(garman_klass_volatility(open_, high, low, close).iloc[-1]) if has_data else 0
        yz_vol = _safe_float(yang_zhang_volatility(open_, high, low, close).iloc[-1]) if has_data else 0

        regime = detect_volatility_regime(returns) if has_data else {"current_regime": "Unknown", "vol_percentile": 0}
        forecast = forecast_volatility(returns, method="ewma", horizon=5) if has_data else {"forecast_vols": [], "current_vol": 0, "long_term_vol": 0, "horizon": 5}
        bollinger = compute_bollinger_bands(close) if has_data else {}

        atr_result = compute_atr(high, low, close)
        current_atr = _safe_float(atr_result) if has_data else 0
        current_close = _safe_float(close.iloc[-1]) if len(close) > 0 else 1

        if isinstance(bollinger, dict):
            bb_bandwidth = _safe_float(bollinger.get("bandwidth", 0))
            bb_pct_b = _safe_float(bollinger.get("percent_b", 0))
        else:
            bb_bandwidth = _safe_float(bollinger["bandwidth"].iloc[-1]) if "bandwidth" in getattr(bollinger, 'columns', []) else 0
            bb_pct_b = _safe_float(bollinger["percent_b"].iloc[-1]) if "percent_b" in getattr(bollinger, 'columns', []) else 0

        forecast_vols = [_safe_float(v) for v in forecast.get("forecast_vols", [])] if isinstance(forecast, dict) else []

        return {
            "ticker": ticker,
            "current_vol": round(ewma_vol, 6),
            "regime": regime.get("current_regime", "Unknown") if isinstance(regime, dict) else str(regime),
            "percentile": regime.get("vol_percentile", 0) if isinstance(regime, dict) else 0,
            "historical_vol": round(hist_vol, 6),
            "ewma_vol": round(ewma_vol, 6),
            "parkinson_vol": round(park_vol, 6),
            "garman_klass_vol": round(gk_vol, 6),
            "yang_zhang_vol": round(yz_vol, 6),
            "atr_pct": round(current_atr / current_close, 6) if current_close > 0 else 0,
            "bb_width": round(bb_bandwidth, 6),
            "bb_pct_b": round(bb_pct_b, 6),
            "forecast": {
                "horizon": forecast.get("horizon", 5) if isinstance(forecast, dict) else 5,
                "forecast_vols": forecast_vols,
                "current_vol": round(_safe_float(forecast.get("current_vol", 0)), 6) if isinstance(forecast, dict) else 0,
                "long_term_vol": round(_safe_float(forecast.get("long_term_vol", 0)), 6) if isinstance(forecast, dict) else 0,
            },
            "position_sizing": {
                "recommended_size": 0.5,
                "vol_scalar": round(ewma_vol / 0.15, 4) if ewma_vol > 0 else 1.0,
                "reasoning": f"Vol regime: {regime.get('current_regime', 'Unknown') if isinstance(regime, dict) else 'Unknown'}",
            },
        }
    except Exception as e:
        return {"error": str(e)}

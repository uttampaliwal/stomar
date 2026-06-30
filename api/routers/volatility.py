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

        hist_vol = float(historical_volatility(returns).iloc[-1]) if len(returns) > 20 else 0
        ewma_vol = float(ewma_volatility(returns).iloc[-1]) if len(returns) > 20 else 0
        park_vol = float(parkinson_volatility(high, low).iloc[-1]) if len(high) > 20 else 0
        gk_vol = float(garman_klass_volatility(open_, high, low, close).iloc[-1]) if len(close) > 20 else 0
        yz_vol = float(yang_zhang_volatility(open_, high, low, close).iloc[-1]) if len(close) > 20 else 0

        regime = detect_volatility_regime(returns)
        forecast = forecast_volatility(returns, method="ewma", horizon=5)
        bollinger = compute_bollinger_bands(close)
        atr = compute_atr(high, low, close)

        current_atr = float(atr.iloc[-1]) if len(atr) > 0 else 0
        current_close = float(close.iloc[-1]) if len(close) > 0 else 1

        return {
            "ticker": ticker,
            "current_vol": ewma_vol,
            "regime": regime.get("current_regime", "Unknown"),
            "percentile": regime.get("vol_percentile", 0),
            "historical_vol": round(hist_vol, 6),
            "ewma_vol": round(ewma_vol, 6),
            "parkinson_vol": round(park_vol, 6),
            "garman_klass_vol": round(gk_vol, 6),
            "yang_zhang_vol": round(yz_vol, 6),
            "atr_pct": round(current_atr / current_close, 6) if current_close > 0 else 0,
            "bb_width": round(float(bollinger["bandwidth"].iloc[-1]), 6) if len(bollinger) > 0 and "bandwidth" in bollinger.columns else 0,
            "bb_pct_b": round(float(bollinger["percent_b"].iloc[-1]), 6) if len(bollinger) > 0 and "percent_b" in bollinger.columns else 0,
            "forecast": {
                "horizon": forecast.get("horizon", 5),
                "forecast_vols": [round(v, 6) for v in forecast.get("forecast_vols", [])],
                "current_vol": round(float(forecast.get("current_vol", 0)), 6),
                "long_term_vol": round(float(forecast.get("long_term_vol", 0)), 6),
            },
            "position_sizing": {
                "recommended_size": 0.5,
                "vol_scalar": round(ewma_vol / 0.15, 4) if ewma_vol > 0 else 1.0,
                "reasoning": f"Vol regime: {regime.get('current_regime', 'Unknown')}",
            },
        }
    except Exception as e:
        return {"error": str(e)}

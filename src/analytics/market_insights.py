from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_recommendation_from_history(frame: pd.DataFrame, ticker: str) -> dict[str, Any]:
    """Create a simple recommendation from actual historical OHLCV data."""
    if frame is None or frame.empty:
        raise ValueError("Historical data frame is empty")

    close = pd.to_numeric(frame["close"], errors="coerce").dropna()
    if close.empty:
        raise ValueError("No close prices available")

    if len(close) < 5:
        raise ValueError("At least 5 data points are required")

    latest = float(close.iloc[-1])
    prev = float(close.iloc[-2])
    trend = (latest - prev) / prev
    sma_fast = close.tail(5).mean()
    sma_slow = close.tail(20).mean()
    trend_strength = (sma_fast - sma_slow) / sma_slow

    returns = close.pct_change().dropna()
    volatility = float(returns.std() * np.sqrt(252))
    momentum = float(returns.iloc[-1] * 100)

    if trend_strength > 0.01 and momentum > 0.3 and volatility < 0.6:
        action = "BUY"
        confidence = min(0.95, 0.65 + abs(trend_strength) * 2.0 + min(0.1, abs(momentum) / 100.0))
        rationale = f"{ticker} shows a positive trend with improving short-term momentum."
        risk_level = "MEDIUM"
    elif trend_strength < -0.01 and momentum < -0.3 and volatility < 0.6:
        action = "SELL"
        confidence = min(0.95, 0.65 + abs(trend_strength) * 2.0 + min(0.1, abs(momentum) / 100.0))
        rationale = f"{ticker} shows weakening momentum and a negative trend."
        risk_level = "MEDIUM"
    else:
        action = "HOLD"
        confidence = 0.55
        rationale = f"{ticker} is mixed; the system is waiting for a cleaner signal."
        risk_level = "LOW"

    return {
        "ticker": ticker,
        "action": action,
        "confidence": round(float(confidence), 2),
        "rationale": rationale,
        "risk_level": risk_level,
        "price": round(latest, 2),
        "trend_pct": round(float(trend * 100), 2),
        "volatility": round(float(volatility), 4),
    }

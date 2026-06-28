import numpy as np
import pandas as pd


def detect_regime(prices: pd.Series, lookback: int = 200) -> dict:
    if len(prices) < lookback:
        return {"regime": "Unknown", "confidence": 0, "indicators": {}}

    close = prices.values
    sma_50 = prices.rolling(50).mean().iloc[-1]
    sma_100 = prices.rolling(100).mean().iloc[-1]
    sma_200 = prices.rolling(200).mean().iloc[-1]
    ema_20 = prices.ewm(span=20).mean().iloc[-1]
    ema_50 = prices.ewm(span=50).mean().iloc[-1]

    current = close[-1]
    rsi = _compute_rsi(prices, 14)

    adx = _compute_adx(prices, 14)

    vol_20 = prices.pct_change().rolling(20).std().iloc[-1] * np.sqrt(252)
    vol_50 = prices.pct_change().rolling(50).std().iloc[-1] * np.sqrt(252)

    momentum_20 = (current / close[-20] - 1) * 100 if len(close) > 20 else 0
    momentum_60 = (current / close[-60] - 1) * 100 if len(close) > 60 else 0

    bull_signals = 0
    bear_signals = 0
    total_signals = 0

    indicators = {}

    total_signals += 1
    if current > sma_200:
        bull_signals += 1
        indicators["Price vs SMA200"] = f"{'Above' if current > sma_200 else 'Below'} (Bull)"
    else:
        bear_signals += 1
        indicators["Price vs SMA200"] = "Below (Bear)"

    total_signals += 1
    if sma_50 > sma_200:
        bull_signals += 1
        indicators["Golden Cross"] = "Yes (Bull)"
    else:
        bear_signals += 1
        indicators["Golden Cross"] = "No (Bear)"

    total_signals += 1
    if ema_20 > ema_50:
        bull_signals += 1
        indicators["EMA Trend"] = "Bullish"
    else:
        bear_signals += 1
        indicators["EMA Trend"] = "Bearish"

    total_signals += 1
    if momentum_20 > 0:
        bull_signals += 1
        indicators["20D Momentum"] = f"{momentum_20:+.1f}% (Bull)"
    else:
        bear_signals += 1
        indicators["20D Momentum"] = f"{momentum_20:+.1f}% (Bear)"

    total_signals += 1
    if momentum_60 > 0:
        bull_signals += 1
        indicators["60D Momentum"] = f"{momentum_60:+.1f}% (Bull)"
    else:
        bear_signals += 1
        indicators["60D Momentum"] = f"{momentum_60:+.1f}% (Bear)"

    total_signals += 1
    if 40 <= rsi <= 60:
        indicators["RSI"] = f"{rsi:.0f} (Neutral)"
    elif rsi > 60:
        bull_signals += 1
        indicators["RSI"] = f"{rsi:.0f} (Bull)"
    else:
        bear_signals += 1
        indicators["RSI"] = f"{rsi:.0f} (Bear)"

    indicators["ADX"] = f"{adx:.0f} ({'Strong' if adx > 25 else 'Weak'} trend)"
    indicators["Volatility (20D)"] = f"{vol_20:.1%}"
    indicators["Volatility (50D)"] = f"{vol_50:.1%}"

    bull_ratio = bull_signals / max(total_signals, 1)

    if bull_ratio > 0.65:
        regime = "Bull"
        confidence = bull_ratio * 100
    elif bull_ratio < 0.35:
        regime = "Bear"
        confidence = (1 - bull_ratio) * 100
    else:
        regime = "Sideways"
        confidence = (1 - abs(bull_ratio - 0.5) * 2) * 100

    return {
        "regime": regime,
        "confidence": round(confidence, 1),
        "bull_signals": bull_signals,
        "bear_signals": bear_signals,
        "total_signals": total_signals,
        "indicators": indicators,
        "recommendation": _get_recommendation(regime, confidence, vol_20, rsi, adx),
    }


def _compute_rsi(prices: pd.Series, period: int = 14) -> float:
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean().iloc[-1]
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean().iloc[-1]
    if loss == 0:
        return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def _compute_adx(prices: pd.Series, period: int = 14) -> float:
    high = prices * 1.01
    low = prices * 0.99
    tr = pd.concat([high - low, abs(high - prices.shift(1)), abs(low - prices.shift(1))], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    up = prices.diff()
    down = -up

    plus_dm = up.where((up > down) & (up > 0), 0)
    minus_dm = down.where((down > up) & (down > 0), 0)

    plus_di = 100 * plus_dm.rolling(period).mean() / atr
    minus_di = 100 * minus_dm.rolling(period).mean() / atr

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
    adx = dx.rolling(period).mean()
    return float(adx.iloc[-1]) if not np.isnan(adx.iloc[-1]) else 20.0


def _get_recommendation(regime: str, confidence: float, vol: float, rsi: float, adx: float) -> dict:
    if regime == "Bull":
        if confidence > 75:
            action = "Aggressive Buy"
            allocation = "80-100% Equity"
        else:
            action = "Gradual Buy"
            allocation = "60-80% Equity"
    elif regime == "Bear":
        if confidence > 75:
            action = "Full Hedge"
            allocation = "20-30% Equity, 70-80% Debt/Gold"
        else:
            action = "Reduce Exposure"
            allocation = "40-60% Equity"
    else:
        action = "Neutral / Hold"
        allocation = "50-70% Equity"

    if rsi > 75:
        action = "Caution: Overbought"
    elif rsi < 25:
        action = "Opportunity: Oversold"

    if vol > 0.40:
        risk = "HIGH - Reduce position sizes"
    elif vol > 0.25:
        risk = "MODERATE"
    else:
        risk = "LOW"

    return {
        "action": action,
        "allocation": allocation,
        "risk_level": risk,
    }

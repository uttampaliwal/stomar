from dataclasses import dataclass
from typing import Optional


@dataclass
class Recommendation:
    action: str
    confidence: float
    rationale: str
    risk_level: str
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


def build_recommendation(price: float, sma_fast: float, sma_slow: float, rsi: float, volatility: float, news_score: float) -> Recommendation:
    """Create a conservative consolidated trading recommendation.

    The engine scores trend, momentum, sentiment, and volatility together.
    It intentionally avoids overconfident signals and defaults to HOLD when the setup is mixed.
    """
    if price <= 0:
        raise ValueError("price must be positive")
    if not 0 <= rsi <= 100:
        raise ValueError("rsi must be between 0 and 100")
    if volatility < 0:
        raise ValueError("volatility must be non-negative")

    trend_signal = 1 if sma_fast > sma_slow else -1 if sma_fast < sma_slow else 0
    momentum_signal = 1 if rsi > 55 else -1 if rsi < 45 else 0
    sentiment_signal = 1 if news_score > 0.2 else -1 if news_score < -0.2 else 0
    volatility_signal = 1 if volatility < 0.03 else 0

    score = sum(
        signal for signal in [trend_signal, momentum_signal, sentiment_signal, volatility_signal] if signal != 0
    )

    if score >= 3 and trend_signal == 1 and momentum_signal == 1 and sentiment_signal == 1:
        action = "BUY"
        confidence = 0.82
        risk_level = "MEDIUM"
        rationale = "The trend, momentum, and sentiment are aligned, and volatility is still moderate."
        stop_loss = round(price * 0.97, 2)
        take_profit = round(price * 1.06, 2)
    elif score <= -3 and trend_signal == -1 and momentum_signal == -1 and sentiment_signal == -1:
        action = "SELL"
        confidence = 0.82
        risk_level = "MEDIUM"
        rationale = "The trend, momentum, and sentiment are weakening, and the setup is clearly bearish."
        stop_loss = round(price * 1.03, 2)
        take_profit = round(price * 0.94, 2)
    else:
        action = "HOLD"
        confidence = 0.6
        risk_level = "LOW"
        rationale = "The signals are mixed or too noisy to justify an entry. Wait for clearer confirmation."
        stop_loss = None
        take_profit = None

    if volatility >= 0.05:
        risk_level = "HIGH"
        confidence = max(0.0, confidence - 0.08)

    return Recommendation(
        action=action,
        confidence=confidence,
        rationale=rationale,
        risk_level=risk_level,
        stop_loss=stop_loss,
        take_profit=take_profit,
    )

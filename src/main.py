import json

from src.trading.recommendation import build_recommendation


def main() -> None:
    recommendation = build_recommendation(
        price=100.0,
        sma_fast=103.0,
        sma_slow=98.0,
        rsi=62.0,
        volatility=0.018,
        news_score=0.35,
    )
    print(json.dumps({
        "action": recommendation.action,
        "confidence": recommendation.confidence,
        "risk_level": recommendation.risk_level,
        "rationale": recommendation.rationale,
        "stop_loss": recommendation.stop_loss,
        "take_profit": recommendation.take_profit,
    }, indent=2))


if __name__ == "__main__":
    main()

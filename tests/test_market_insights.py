import pandas as pd

from src.analytics.market_insights import build_recommendation_from_history


def _make_frame(values):
    return pd.DataFrame(
        {"close": values, "volume": [1000 + i for i in range(len(values))]},
        index=pd.date_range("2024-01-01", periods=len(values), freq="D"),
    )


def test_bullish_frame_returns_buy_signal():
    frame = _make_frame([100, 102, 105, 108, 111, 114, 118, 121, 125])
    result = build_recommendation_from_history(frame, ticker="TEST.NS")
    assert result["action"] == "BUY"
    assert result["confidence"] >= 0.6


def test_bearish_frame_returns_sell_signal():
    frame = _make_frame([125, 121, 118, 114, 111, 108, 105, 102, 100])
    result = build_recommendation_from_history(frame, ticker="TEST.NS")
    assert result["action"] == "SELL"
    assert result["confidence"] >= 0.6


def test_flat_frame_returns_hold_signal():
    frame = _make_frame([100, 100.2, 100.3, 100.1, 100.4, 100.2, 100.5, 100.3, 100.4])
    result = build_recommendation_from_history(frame, ticker="TEST.NS")
    assert result["action"] == "HOLD"

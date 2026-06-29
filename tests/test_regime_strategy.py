"""Tests for src/regime_strategy.py."""

import numpy as np
import pandas as pd
import pytest

from src.regime_strategy import (
    get_regime_allocation,
    regime_adjusted_position_size,
    generate_regime_signals,
    backtest_regime_strategy,
)


def _make_df(n=100, start="2024-01-01"):
    dates = pd.bdate_range(start, periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


# --- get_regime_allocation ---

def test_bull_allocation():
    alloc = get_regime_allocation("Bull", confidence=0.8)
    assert alloc["equity_pct"] == 0.80
    assert alloc["cash_pct"] == 0.20
    assert alloc["strategy"] == "momentum"


def test_bear_allocation():
    alloc = get_regime_allocation("Bear", confidence=0.8)
    assert alloc["equity_pct"] == 0.20
    assert alloc["cash_pct"] == 0.80
    assert alloc["strategy"] == "defensive"


def test_sideways_allocation():
    alloc = get_regime_allocation("Sideways", confidence=0.8)
    assert alloc["equity_pct"] == 0.50
    assert alloc["cash_pct"] == 0.50
    assert alloc["strategy"] == "mean_reversion"


def test_unknown_regime_defaults_to_sideways():
    alloc = get_regime_allocation("Unknown", confidence=0.5)
    assert alloc["equity_pct"] == 0.50


def test_low_confidence_reduces_equity():
    alloc_normal = get_regime_allocation("Bull", confidence=0.8)
    alloc_low = get_regime_allocation("Bull", confidence=0.2)
    assert alloc_low["equity_pct"] < alloc_normal["equity_pct"]
    assert "low confidence" in alloc_low["reasoning"]


# --- regime_adjusted_position_size ---

def test_bull_multiplier_increases_size():
    result = regime_adjusted_position_size(0.10, "Bull")
    assert result["position_size_pct"] > 0.10
    assert result["regime_multiplier"] == 1.2


def test_bear_multiplier_decreases_size():
    result = regime_adjusted_position_size(0.10, "Bear")
    assert result["position_size_pct"] < 0.10
    assert result["regime_multiplier"] == 0.5


def test_sideways_multiplier_neutral():
    result = regime_adjusted_position_size(0.10, "Sideways")
    assert result["position_size_pct"] == pytest.approx(0.10, abs=0.001)
    assert result["regime_multiplier"] == 1.0


def test_vol_adjustment_scales_by_volatility():
    result_low_vol = regime_adjusted_position_size(0.10, "Sideways", current_vol=0.10)
    result_high_vol = regime_adjusted_position_size(0.10, "Sideways", current_vol=0.40)
    assert result_low_vol["position_size_pct"] > result_high_vol["position_size_pct"]


def test_position_capped_at_25pct():
    result = regime_adjusted_position_size(0.50, "Bull")
    assert result["position_size_pct"] <= 0.25


def test_vol_scalar_bounded():
    result = regime_adjusted_position_size(0.10, "Sideways", current_vol=0.01)
    assert result["position_size_pct"] <= 0.25


# --- generate_regime_signals ---

def test_generate_signals_returns_regime():
    df = _make_df(100)
    result = generate_regime_signals(df)
    assert "regime" in result
    assert "confidence" in result
    assert "allocation" in result
    assert "signals" in result


def test_generate_signals_length_matches_df():
    df = _make_df(100)
    result = generate_regime_signals(df)
    assert len(result["signals"]) == 99  # n-1 signals


def test_signals_have_required_keys():
    df = _make_df(100)
    result = generate_regime_signals(df)
    for sig in result["signals"]:
        assert "date" in sig
        assert "signal" in sig
        assert "price" in sig


def test_bear_regime_all_zero_signals():
    df = _make_df(100)
    # Detect what regime we get — if Bear, signals should all be 0
    result = generate_regime_signals(df)
    if result["regime"] == "Bear":
        assert all(s["signal"] == 0 for s in result["signals"])


# --- backtest_regime_strategy ---

def test_backtest_returns_required_keys():
    df = _make_df(100)
    result = backtest_regime_strategy(df)
    required = [
        "regime", "equity_allocation", "strategy_return",
        "strategy_annualized", "strategy_sharpe",
        "buy_hold_return", "buy_hold_annualized",
        "excess_return", "n_days",
    ]
    for key in required:
        assert key in result


def test_backtest_n_days_matches_returns():
    df = _make_df(100)
    result = backtest_regime_strategy(df)
    assert result["n_days"] == 99  # pct_change loses 1 row


def test_backtest_equity_allocation_is_float():
    df = _make_df(100)
    result = backtest_regime_strategy(df)
    assert isinstance(result["equity_allocation"], float)
    assert 0 <= result["equity_allocation"] <= 1


def test_backtest_with_precomputed_regime():
    df = _make_df(100)
    regime_result = {"regime": "Bull", "confidence": 0.9}
    result = backtest_regime_strategy(df, regime_result=regime_result)
    assert result["regime"] == "Bull"
    assert result["equity_allocation"] == 0.80

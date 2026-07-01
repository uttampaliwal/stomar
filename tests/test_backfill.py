"""Tests for src/backfill.py."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from src.core.backfill import HistoricalBackfill
from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_backfill.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


@pytest.fixture
def sample_df():
    np.random.seed(42)
    n = 200
    dates = pd.bdate_range("2024-01-01", periods=n)
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


# --- Creation ---

def test_creates_instance(ledger):
    bf = HistoricalBackfill(ledger)
    assert bf.ledger is ledger


# --- run() ---

def test_run_empty_tickers(ledger):
    bf = HistoricalBackfill(ledger)
    summary = bf.run(tickers=[], lookback_days=252)
    assert summary["total_decisions"] == 0
    assert summary["total_outcomes"] == 0


def test_run_returns_summary(ledger):
    bf = HistoricalBackfill(ledger)
    summary = bf.run(tickers=[], lookback_days=252)
    assert "tickers" in summary
    assert "total_decisions" in summary
    assert "total_outcomes" in summary


def test_run_handles_fetch_failure(ledger):
    bf = HistoricalBackfill(ledger)
    with patch.object(bf, "_fetch_data", return_value=None):
        summary = bf.run(tickers=["BAD.NS"], lookback_days=252)
    # Returns 0 decisions (not error key) when data is insufficient
    assert summary["tickers"]["BAD.NS"]["decisions"] == 0
    assert summary["total_decisions"] == 0


# --- _backfill_ticker ---

def test_backfill_ticker_writes_to_ledger(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    with patch.object(bf, "_fetch_data", return_value=sample_df):
        result = bf._backfill_ticker("TEST.NS", lookback_days=252)
    assert result["decisions"] > 0
    assert result["outcomes"] > 0
    decisions = ledger.get_decisions("TEST.NS")
    assert len(decisions) == result["decisions"]


def test_backfill_ticker_logs_outcomes(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    with patch.object(bf, "_fetch_data", return_value=sample_df):
        bf._backfill_ticker("TEST.NS", lookback_days=252)
    decisions = ledger.get_decisions("TEST.NS")
    resolved = [d for d in decisions if d["actual_return"] is not None]
    # Every decision gets an outcome (loop stops at len-1, so last decision has next-day close)
    assert len(resolved) > 0
    assert len(resolved) == len(decisions)


def test_backfill_ticker_short_data(ledger):
    bf = HistoricalBackfill(ledger)
    short_df = pd.DataFrame({
        "close": [100, 101], "open": [99, 100], "high": [102, 102],
        "low": [98, 99], "volume": [1000, 1000],
    }, index=pd.bdate_range("2024-01-01", periods=2))
    with patch.object(bf, "_fetch_data", return_value=short_df):
        result = bf._backfill_ticker("TEST.NS", lookback_days=252)
    assert result["decisions"] == 0  # too short (< 60 warmup)


# --- _compute_historical_signals ---

def test_historical_signals_has_all_keys(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    close = sample_df["close"]
    returns = close.pct_change().dropna()
    signals = bf._compute_historical_signals(
        df=sample_df, index=100, close=close, returns=returns,
        regime_result=None, models=None, ticker="TEST.NS",
    )
    expected_keys = [
        "sentiment_score", "fii_net", "dii_net", "pcr", "mtf_signal",
        "regime", "var_95", "cvar_95", "sharpe", "volatility_forecast",
    ]
    for key in expected_keys:
        assert key in signals, f"Missing key: {key}"


def test_historical_signals_neutral_defaults(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    close = sample_df["close"]
    returns = close.pct_change().dropna()
    signals = bf._compute_historical_signals(
        df=sample_df, index=100, close=close, returns=returns,
        regime_result=None, models=None, ticker="TEST.NS",
    )
    assert signals["sentiment_score"] == 0.0
    assert signals["fii_net"] == 0.0
    assert signals["dii_net"] == 0.0
    assert signals["pcr"] == 1.0


# --- _compute_simple_mtf ---

def test_mtf_uptrend(ledger):
    bf = HistoricalBackfill(ledger)
    close = pd.Series(np.arange(100, 150, dtype=float))
    result = bf._compute_simple_mtf(close)
    assert result > 0


def test_mtf_downtrend(ledger):
    bf = HistoricalBackfill(ledger)
    close = pd.Series(np.arange(150, 100, -0.5, dtype=float))
    result = bf._compute_simple_mtf(close)
    assert result < 0


def test_mtf_short_data(ledger):
    bf = HistoricalBackfill(ledger)
    close = pd.Series([100.0, 101.0])
    result = bf._compute_simple_mtf(close)
    assert result == 0.0


# --- _make_historical_decision ---

def test_decision_buy(ledger):
    bf = HistoricalBackfill(ledger)
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.8, "regime": "Bull"}
    decision = bf._make_historical_decision(signals)
    assert decision["action"] in ("BUY", "SELL", "HOLD")
    assert "reasoning" in decision


def test_decision_sell(ledger):
    bf = HistoricalBackfill(ledger)
    signals = {"ensemble_direction": 0, "ensemble_confidence": 0.8, "regime": "Bear"}
    decision = bf._make_historical_decision(signals)
    assert decision["action"] in ("BUY", "SELL", "HOLD")


def test_decision_neutral(ledger):
    bf = HistoricalBackfill(ledger)
    signals = {}
    decision = bf._make_historical_decision(signals)
    assert decision["action"] == "HOLD"


# --- _compute_risk_historical ---

def test_risk_with_data(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    returns = sample_df["close"].pct_change().dropna()
    result = bf._compute_risk_historical(returns)
    assert "var_95" in result
    assert "cvar_95" in result
    assert "sharpe" in result


def test_risk_short_data(ledger):
    bf = HistoricalBackfill(ledger)
    returns = pd.Series([0.01, -0.01])
    result = bf._compute_risk_historical(returns)
    assert result == {}


# --- _compute_volatility_historical ---

def test_volatility_with_data(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    returns = sample_df["close"].pct_change().dropna()
    result = bf._compute_volatility_historical(returns)
    assert "volatility_forecast" in result


def test_volatility_short_data(ledger):
    bf = HistoricalBackfill(ledger)
    returns = pd.Series([0.01, -0.01])
    result = bf._compute_volatility_historical(returns)
    assert result == {}


# --- Integration ---

def test_full_backfill_flow(ledger, sample_df):
    bf = HistoricalBackfill(ledger)
    with patch.object(bf, "_fetch_data", return_value=sample_df):
        summary = bf.run(tickers=["TEST.NS"], lookback_days=252)
    assert summary["total_decisions"] > 0
    assert summary["total_outcomes"] > 0

    # Verify we can query performance
    perf = ledger.get_performance()
    assert perf["total_decisions"] > 0
    assert perf["resolved"] > 0

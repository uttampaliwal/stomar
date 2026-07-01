"""Tests for src/orchestrator.py."""

import os
import tempfile
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.signals.orchestrator import DailyOrchestrator
from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_orch.db")
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


def _mock_meta_controller():
    mc = MagicMock()
    mc.decide.return_value = {
        "action": "HOLD",
        "position_size": 0.0,
        "confidence": 0.5,
        "reasoning": "Mock decision",
    }
    return mc


# --- Basic creation ---

def test_creates_instance(ledger):
    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
    assert orch.tickers == ["TEST.NS"]
    assert orch.ledger is ledger


def test_creates_with_meta_controller(ledger):
    mc = _mock_meta_controller()
    orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger, meta_controller=mc)
    assert orch.meta_controller is mc


# --- run() ---

def test_run_returns_summary(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    summary = orch.run()
    assert "date" in summary
    assert "decisions" in summary
    assert "errors" in summary


def test_run_logs_to_ledger(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    orch.run(dry_run=True)
    decisions = ledger.get_decisions()
    assert len(decisions) == 0  # dry run shouldn't log


def test_run_dry_run_doesnt_write(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    orch.run(dry_run=True)
    assert len(ledger.get_decisions()) == 0


def test_run_handles_empty_tickers(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    summary = orch.run()
    assert len(summary["decisions"]) == 0
    assert len(summary["errors"]) == 0


# --- _process_ticker ---

def test_process_ticker_collects_signals(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    with patch.object(orch, "_fetch_data") as mock_fetch, \
         patch.object(orch, "_run_ensemble", return_value={"ensemble_direction": 1, "ensemble_confidence": 0.7}), \
         patch.object(orch, "_run_sentiment", return_value={"sentiment_score": 0.3}), \
         patch.object(orch, "_run_flow", return_value={"fii_net": 500}), \
         patch.object(orch, "_run_pcr", return_value={"pcr": 1.1}), \
         patch.object(orch, "_run_mtf", return_value={"mtf_signal": 0.5}), \
         patch.object(orch, "_run_regime", return_value={"regime": "Bull", "regime_confidence": 0.8}), \
         patch.object(orch, "_run_risk", return_value={"var_95": -0.02}), \
         patch.object(orch, "_run_volatility", return_value={"volatility_forecast": 0.18}), \
         patch.object(orch, "_run_fundamentals", return_value={"fundamental_score": 70}):
        mock_fetch.return_value = pd.DataFrame({
            "close": np.random.randn(100) + 100,
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 101,
            "low": np.random.randn(100) + 99,
            "volume": np.random.randint(1000, 10000, 100).astype(float),
        }, index=pd.bdate_range("2024-01-01", periods=100))
        result = orch._process_ticker("TEST.NS", "2025-01-15", dry_run=True)
    assert result["ticker"] == "TEST.NS"
    assert "action" in result
    assert "signals" in result


def test_process_ticker_handles_fetch_failure(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    with patch.object(orch, "_fetch_data", return_value=None):
        result = orch._process_ticker("BAD.NS", "2025-01-15", dry_run=True)
    assert result["action"] == "HOLD"  # default when no data


def test_process_ticker_logs_to_ledger(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    with patch.object(orch, "_fetch_data") as mock_fetch, \
         patch.object(orch, "_run_ensemble", return_value={}), \
         patch.object(orch, "_run_sentiment", return_value={}), \
         patch.object(orch, "_run_flow", return_value={}), \
         patch.object(orch, "_run_pcr", return_value={}), \
         patch.object(orch, "_run_mtf", return_value={}), \
         patch.object(orch, "_run_regime", return_value={}), \
         patch.object(orch, "_run_risk", return_value={}), \
         patch.object(orch, "_run_volatility", return_value={}), \
         patch.object(orch, "_run_fundamentals", return_value={}):
        mock_fetch.return_value = pd.DataFrame({
            "close": np.random.randn(100) + 100,
            "open": np.random.randn(100) + 100,
            "high": np.random.randn(100) + 101,
            "low": np.random.randn(100) + 99,
            "volume": np.random.randint(1000, 10000, 100).astype(float),
        }, index=pd.bdate_range("2024-01-01", periods=100))
        orch._process_ticker("TEST.NS", "2025-01-15", dry_run=False)
    decisions = ledger.get_decisions()
    assert len(decisions) == 1
    assert decisions[0]["ticker"] == "TEST.NS"


# --- _make_decision ---

def test_make_decision_uses_meta_controller(ledger):
    mc = _mock_meta_controller()
    orch = DailyOrchestrator(tickers=[], ledger=ledger, meta_controller=mc)
    result = orch._make_decision({"ensemble_direction": 1})
    mc.decide.assert_called_once_with({"ensemble_direction": 1})
    assert result["action"] == "HOLD"


def test_make_decision_default_fallback(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    result = orch._make_decision({"ensemble_direction": 1, "ensemble_confidence": 0.7})
    assert result["action"] == "BUY"


def test_make_decision_default_hold(ledger):
    orch = DailyOrchestrator(tickers=[], ledger=ledger)
    result = orch._make_decision({"ensemble_direction": None})
    assert result["action"] == "HOLD"


# --- run() with multiple tickers ---

def test_run_multiple_tickers(ledger):
    orch = DailyOrchestrator(tickers=["A.NS", "B.NS"], ledger=ledger)
    with patch.object(orch, "_process_ticker") as mock_process:
        mock_process.return_value = {"ticker": "X.NS", "action": "HOLD", "position_size": 0, "confidence": 0.5, "reasoning": "test"}
        summary = orch.run()
    assert len(summary["decisions"]) == 2


def test_run_catches_errors(ledger):
    orch = DailyOrchestrator(tickers=["GOOD.NS", "BAD.NS"], ledger=ledger)
    with patch.object(orch, "_process_ticker") as mock_process:
        def side_effect(ticker, date, dry_run):
            if ticker == "BAD.NS":
                raise ValueError("Something went wrong")
            return {"ticker": ticker, "action": "HOLD", "position_size": 0, "confidence": 0.5, "reasoning": "ok"}
        mock_process.side_effect = side_effect
        summary = orch.run()
    assert len(summary["decisions"]) == 1
    assert len(summary["errors"]) == 1
    assert summary["errors"][0]["ticker"] == "BAD.NS"

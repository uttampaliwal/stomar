"""Tests for daily outcome resolution (P3.3) in the orchestrator."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.signals.orchestrator import DailyOrchestrator
from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_outcome.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


def _signals():
    return {
        "ensemble_direction": 1,
        "ensemble_confidence": 0.7,
        "sentiment_score": 0.3,
        "fii_net": 100.0,
        "dii_net": 50.0,
        "pcr": 1.1,
        "mtf_signal": 0.5,
        "regime": "Bull",
        "regime_confidence": 0.7,
        "var_95": -0.02,
        "cvar_95": -0.03,
        "sharpe": 1.0,
        "volatility_forecast": 0.15,
        "fundamental_score": 60.0,
    }


def _price_df(dates):
    n = len(dates)
    close = 100 + np.cumsum(np.random.default_rng(42).normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": np.full(n, 1_000_000.0),
    }, index=dates)


class TestResolvePendingOutcomes:
    def test_resolves_previous_day_decision(self, ledger):
        dates = pd.bdate_range("2025-01-01", periods=5)
        did = ledger.log_decision(
            date=str(dates[2].date()), ticker="TEST.NS", signals=_signals(),
            action="BUY", position_size=0.05, confidence=0.7, source="live",
        )
        orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
        with patch.object(orch, "_fetch_data", return_value=_price_df(dates)):
            summary = orch.resolve_pending_outcomes()

        assert summary["resolved"] == 1
        row = ledger.get_decisions()[0]
        assert row["actual_return"] is not None
        assert row["actual_direction"] in (0, 1)
        assert row["id"] == did

    def test_skips_decision_with_no_next_day(self, ledger):
        dates = pd.bdate_range("2025-01-01", periods=3)
        ledger.log_decision(
            date=str(dates[-1].date()), ticker="TEST.NS", signals=_signals(),
            action="BUY", position_size=0.05, confidence=0.7, source="live",
        )
        orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
        with patch.object(orch, "_fetch_data", return_value=_price_df(dates)):
            summary = orch.resolve_pending_outcomes()
        assert summary["resolved"] == 0
        assert summary["failed"] == 1

    def test_backfill_decisions_are_skipped(self, ledger):
        dates = pd.bdate_range("2025-01-01", periods=5)
        ledger.log_decision(
            date=str(dates[2].date()), ticker="TEST.NS", signals=_signals(),
            action="BUY", position_size=0.05, confidence=0.7, source="backfill",
        )
        orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
        with patch.object(orch, "_fetch_data", return_value=_price_df(dates)):
            summary = orch.resolve_pending_outcomes()
        assert summary["resolved"] == 0
        assert summary["pending"] == 1

    def test_resolve_lands_on_next_trading_day(self, ledger):
        dates = pd.bdate_range("2025-01-01", periods=4)
        df = _price_df(dates)
        entry_close = float(df["close"].iloc[2])
        next_close = float(df["close"].iloc[3])
        expected_ret = (next_close - entry_close) / entry_close
        did = ledger.log_decision(
            date=str(dates[2].date()), ticker="TEST.NS", signals=_signals(),
            action="BUY", position_size=0.05, confidence=0.7, source="live",
        )
        orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
        with patch.object(orch, "_fetch_data", return_value=df):
            orch.resolve_pending_outcomes()
        row = ledger.get_decisions()[0]
        assert row["actual_return"] == pytest.approx(expected_ret)
        assert row["actual_direction"] == (1 if expected_ret > 0 else 0)
        assert row["id"] == did


class TestRunWiring:
    def test_run_resolves_outcomes_when_requested(self, ledger):
        dates = pd.bdate_range("2025-01-01", periods=6)
        ledger.log_decision(
            date=str(dates[3].date()), ticker="TEST.NS", signals=_signals(),
            action="BUY", position_size=0.05, confidence=0.7, source="live",
        )
        orch = DailyOrchestrator(tickers=["TEST.NS"], ledger=ledger)
        hold = {
            "action": "HOLD", "position_size": 0.0, "confidence": 0.4,
            "reasoning": "test",
        }
        with patch.object(orch, "_fetch_data", return_value=_price_df(dates)), \
             patch.object(orch, "_run_flow", return_value={}), \
             patch.object(orch, "_run_pcr", return_value={}), \
             patch.object(orch, "_make_decision", return_value=hold), \
             patch.object(orch, "_collect_signals",
                          return_value={**_signals(), "current_price": 100.0}):
            summary = orch.run(date=str(dates[4].date()), dry_run=False,
                               resolve_outcomes=True)

        assert summary["outcomes"]["resolved"] == 1

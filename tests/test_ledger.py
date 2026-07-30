"""Tests for src/ledger.py."""

import os
import json
import tempfile
import pytest

from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    """Create a temporary ledger for testing."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_ledger.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


@pytest.fixture
def sample_signals():
    return {
        "ensemble_direction": 1,
        "ensemble_confidence": 0.72,
        "sentiment_score": 0.35,
        "fii_net": 1500.0,
        "dii_net": -500.0,
        "pcr": 1.15,
        "max_pain": 24500.0,
        "mtf_signal": 0.6,
        "regime": "Bull",
        "regime_confidence": 0.8,
        "var_95": -0.023,
        "cvar_95": -0.035,
        "sharpe": 1.2,
        "volatility_forecast": 0.18,
        "fundamental_score": 72.5,
    }


# --- Table creation ---

def test_creates_db(ledger):
    tables = ledger.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    table_names = {t["name"] for t in tables}
    assert "decisions" in table_names
    assert "paper_trades" in table_names
    assert "portfolio_snapshots" in table_names


def test_creates_indexes(ledger):
    indexes = ledger.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    index_names = {i["name"] for i in indexes}
    assert "idx_decisions_date" in index_names
    assert "idx_decisions_ticker" in index_names


def test_creates_schema_version(ledger):
    row = ledger.conn.execute("SELECT version FROM schema_version").fetchone()
    assert row is not None
    assert row["version"] == 1


def test_reopen_existing_db_preserves_version(ledger, sample_signals):
    ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    db_path = ledger.db_path
    ledger.close()
    lg2 = Ledger(db_path)
    row = lg2.conn.execute("SELECT version FROM schema_version").fetchone()
    assert row["version"] == 1
    decisions = lg2.get_decisions()
    assert len(decisions) == 1
    lg2.close()


# --- log_decision ---

def test_log_decision_returns_id(ledger, sample_signals):
    decision_id = ledger.log_decision(
        date="2025-01-15", ticker="RELIANCE.NS",
        signals=sample_signals, action="BUY",
        position_size=0.05, confidence=0.7,
    )
    assert isinstance(decision_id, int)
    assert decision_id > 0


def test_log_decision_stores_signals(ledger, sample_signals):
    decision_id = ledger.log_decision(
        date="2025-01-15", ticker="RELIANCE.NS",
        signals=sample_signals, action="BUY",
        position_size=0.05, confidence=0.7,
    )
    row = ledger.conn.execute(
        "SELECT * FROM decisions WHERE id = ?", (decision_id,)
    ).fetchone()
    assert row["ticker"] == "RELIANCE.NS"
    assert row["action"] == "BUY"
    assert row["ensemble_confidence"] == 0.72
    assert row["sentiment_score"] == 0.35
    assert row["regime"] == "Bull"


def test_log_decision_increments_id(ledger, sample_signals):
    id1 = ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    id2 = ledger.log_decision("2025-01-15", "TCS.NS", sample_signals, "SELL", 0.03, 0.6)
    assert id2 == id1 + 1


# --- log_trade ---

def test_log_trade_links_to_decision(ledger, sample_signals):
    decision_id = ledger.log_decision(
        "2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7
    )
    ledger.log_trade(decision_id, "RELIANCE.NS", "BUY", 10, 2500.0, 2.5, 1.5)
    trades = ledger.get_trades("RELIANCE.NS")
    assert len(trades) == 1
    assert trades[0]["decision_id"] == decision_id
    assert trades[0]["side"] == "BUY"
    assert trades[0]["total_cost"] == 4.0


def test_log_trade_stores_fields(ledger, sample_signals):
    decision_id = ledger.log_decision(
        "2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7
    )
    ledger.log_trade(decision_id, "RELIANCE.NS", "BUY", 10, 2500.0, 2.5, 1.5)
    row = ledger.conn.execute("SELECT * FROM paper_trades").fetchone()
    assert row["quantity"] == 10
    assert row["price"] == 2500.0
    assert row["slippage"] == 2.5
    assert row["costs"] == 1.5


# --- log_outcome ---

def test_log_outcome_updates_decision(ledger, sample_signals):
    decision_id = ledger.log_decision(
        "2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7
    )
    ledger.log_outcome(decision_id, actual_return=0.012, actual_direction=1)
    row = ledger.conn.execute(
        "SELECT actual_return, actual_direction, correct FROM decisions WHERE id = ?",
        (decision_id,)
    ).fetchone()
    assert row["actual_return"] == 0.012
    assert row["actual_direction"] == 1
    assert row["correct"] == 1  # ensemble_direction was 1, actual was 1


def test_log_outcome_marks_incorrect(ledger, sample_signals):
    decision_id = ledger.log_decision(
        "2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7
    )
    ledger.log_outcome(decision_id, actual_return=-0.008, actual_direction=0)
    row = ledger.conn.execute(
        "SELECT correct FROM decisions WHERE id = ?", (decision_id,)
    ).fetchone()
    assert row["correct"] == 0


def test_log_outcome_nonexistent_decision(ledger):
    ledger.log_outcome(999, actual_return=0.01, actual_direction=1)


# --- log_snapshot ---

def test_log_snapshot_stores_data(ledger):
    ledger.log_snapshot("2025-01-15", 100000.0, 20000.0, {"RELIANCE.NS": 80000.0})
    snapshots = ledger.get_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["total_value"] == 100000.0
    assert snapshots[0]["cash"] == 20000.0
    holdings = json.loads(snapshots[0]["holdings_json"])
    assert holdings["RELIANCE.NS"] == 80000.0


def test_log_snapshot_replaces_same_date(ledger):
    ledger.log_snapshot("2025-01-15", 100000.0, 20000.0, {})
    ledger.log_snapshot("2025-01-15", 105000.0, 25000.0, {})
    snapshots = ledger.get_snapshots()
    assert len(snapshots) == 1
    assert snapshots[0]["total_value"] == 105000.0


# --- get_decisions ---

def test_get_decisions_filters_by_ticker(ledger, sample_signals):
    ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    ledger.log_decision("2025-01-15", "TCS.NS", sample_signals, "SELL", 0.03, 0.6)
    r = ledger.get_decisions(ticker="RELIANCE.NS")
    assert len(r) == 1
    assert r[0]["ticker"] == "RELIANCE.NS"


def test_get_decisions_filters_by_date(ledger, sample_signals):
    ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    ledger.log_decision("2025-01-20", "RELIANCE.NS", sample_signals, "SELL", 0.03, 0.6)
    r = ledger.get_decisions(start_date="2025-01-18")
    assert len(r) == 1
    assert r[0]["date"] == "2025-01-20"


# --- get_performance ---

def test_get_performance_empty(ledger):
    perf = ledger.get_performance()
    assert perf["total_decisions"] == 0
    assert perf["accuracy"] is None


def test_get_performance_with_data(ledger, sample_signals):
    id1 = ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    id2 = ledger.log_decision("2025-01-15", "TCS.NS", sample_signals, "HOLD", 0.0, 0.5)
    ledger.log_outcome(id1, 0.012, 1)  # correct (BUY + price up)
    ledger.log_outcome(id2, -0.005, 0)  # correct (HOLD is always correct - no trade = no loss)
    perf = ledger.get_performance()
    assert perf["total_decisions"] == 2
    assert perf["resolved"] == 2
    assert perf["correct_predictions"] == 2


# --- get_signal_accuracy ---

def test_get_signal_accuracy_empty(ledger):
    acc = ledger.get_signal_accuracy()
    assert acc == {}


def test_get_signal_accuracy_computes(ledger, sample_signals):
    id1 = ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    ledger.log_outcome(id1, 0.012, 1)
    acc = ledger.get_signal_accuracy()
    assert "ensemble_direction" in acc
    assert acc["ensemble_direction"] == 1.0  # direction was 1, actual was 1


# --- get_daily_pnl ---

def test_get_daily_pnl_empty(ledger):
    pnl = ledger.get_daily_pnl()
    assert pnl == []


def test_get_daily_pnl_computes(ledger):
    ledger.log_snapshot("2025-01-15", 100000.0, 20000.0, {})
    ledger.log_snapshot("2025-01-16", 102000.0, 22000.0, {})
    pnl = ledger.get_daily_pnl()
    assert len(pnl) == 2
    assert pnl[0]["daily_pnl"] == 0.0
    assert pnl[1]["daily_pnl"] == 2000.0
    assert pnl[1]["daily_return"] == pytest.approx(0.02, abs=0.001)


# --- get_trades ---

def test_get_trades_all(ledger, sample_signals):
    decision_id = ledger.log_decision("2025-01-15", "RELIANCE.NS", sample_signals, "BUY", 0.05, 0.7)
    ledger.log_trade(decision_id, "RELIANCE.NS", "BUY", 10, 2500.0)
    ledger.log_trade(decision_id, "TCS.NS", "BUY", 5, 3500.0)
    trades = ledger.get_trades()
    assert len(trades) == 2


# --- close ---

def test_close_prevents_further_queries(ledger):
    ledger.close()
    with pytest.raises(Exception):
        ledger.get_decisions()

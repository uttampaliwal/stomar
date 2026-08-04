"""Tests for src/core/readiness_check.py (P3.5 gate)."""

import json
import os
import tempfile
from datetime import datetime, timedelta

import pytest

from src.trading.ledger import Ledger
from src.core import readiness_check
from src.core.readiness_check import check_readiness


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_readiness.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


def _signals():
    return {
        "ensemble_direction": 1,
        "ensemble_confidence": 0.72,
        "sentiment_score": 0.35,
        "fii_net": 1500.0,
        "dii_net": -500.0,
        "pcr": 1.15,
        "mtf_signal": 0.6,
        "regime": "Bull",
        "regime_confidence": 0.8,
        "var_95": -0.023,
        "cvar_95": -0.035,
        "sharpe": 1.2,
        "volatility_forecast": 0.18,
        "fundamental_score": 72.5,
    }


def _seed_live_decisions(ledger, n_days=65, n_per_day=1, correct_ratio=0.6,
                         actions=("BUY", "SELL", "HOLD")):
    """Seed resolved live decisions across consecutive business days."""
    import pandas as pd
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    did = 0
    for i, d in enumerate(dates):
        for _ in range(n_per_day):
            did += 1
            action = actions[i % len(actions)]
            conf = 0.5 + (i % 5) * 0.1
            decision_id = ledger.log_decision(
                date=str(d.date()), ticker="TEST.NS", signals=_signals(),
                action=action, position_size=0.05 if action != "HOLD" else 0.0,
                confidence=conf, source="live",
            )
            # Deterministic pseudo-random spread: (did*7 % 100) < ratio*100
            # gives ~ratio correctness in ANY window, avoiding cyclic bias.
            correct = ((did * 7) % 100) < (100 * correct_ratio)
            ret = 0.012 if correct else -0.012
            if action == "HOLD":
                ret = 0.001  # sideways
                correct = True
            ledger.log_outcome(decision_id, ret, 1 if ret > 0 else 0)
    return dates


def _seed_snapshots(ledger, n=65, start_value=100_000):
    import pandas as pd
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    value = start_value
    for d in dates:
        value = value * (1 + 0.001)
        ledger.log_snapshot(str(d.date()), round(value, 2), 5000, {})


class TestReadinessGates:
    def test_all_gates_pass_with_full_evidence(self, ledger, tmp_path, monkeypatch):
        _seed_live_decisions(ledger, n_days=65, correct_ratio=0.7)
        _seed_snapshots(ledger, n=65)

        # Fresh models + no drift alerts
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        for t in ["TEST.NS"]:
            model = os.path.join(str(tmp_path), f"{t.replace('.', '_')}_models.pkl")
            with open(model, "w") as f:
                f.write("x")
            os.utime(model, (datetime.now().timestamp(), datetime.now().timestamp()))

        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["ready"] is True, report
        assert all(g["passed"] for g in report["gates"].values())

    def test_fails_without_paper_period(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        for t in ["TEST.NS"]:
            model = os.path.join(str(tmp_path), f"{t.replace('.', '_')}_models.pkl")
            with open(model, "w") as f:
                f.write("x")
        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["ready"] is False
        assert report["gates"]["paper_period"]["passed"] is False
        assert report["gates"]["accuracy"]["passed"] is False

    def test_fails_when_accuracy_below_threshold(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        model = os.path.join(str(tmp_path), "TEST_NS_models.pkl")
        with open(model, "w") as f:
            f.write("x")
        _seed_snapshots(ledger, n=65)
        _seed_live_decisions(ledger, n_days=65, correct_ratio=0.4)
        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["ready"] is False
        assert report["gates"]["accuracy"]["passed"] is False

    def test_fails_on_stale_models(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        model = os.path.join(str(tmp_path), "TEST_NS_models.pkl")
        with open(model, "w") as f:
            f.write("x")
        old = datetime.now() - timedelta(days=90)
        os.utime(model, (old.timestamp(), old.timestamp()))
        _seed_snapshots(ledger, n=65)
        _seed_live_decisions(ledger, n_days=65, correct_ratio=0.7)
        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["ready"] is False
        assert report["gates"]["models_fresh"]["passed"] is False

    def test_fails_on_critical_drift_alerts(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        model = os.path.join(str(tmp_path), "TEST_NS_models.pkl")
        with open(model, "w") as f:
            f.write("x")
        _seed_snapshots(ledger, n=65)
        _seed_live_decisions(ledger, n_days=65, correct_ratio=0.7)

        alert_dir = tmp_path / "alerts"
        alert_dir.mkdir(exist_ok=True)
        (alert_dir / "TEST_NS_alerts.json").write_text(json.dumps([{
            "severity": "critical",
            "timestamp": datetime.now().isoformat(),
            "message": "feature drift",
        }]))
        monkeypatch.setattr(readiness_check, "MONITORING_DIR", str(alert_dir))

        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["ready"] is False
        assert report["gates"]["drift"]["passed"] is False

    def test_drawdown_gate_detects_deep_equity_drop(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        model = os.path.join(str(tmp_path), "TEST_NS_models.pkl")
        with open(model, "w") as f:
            f.write("x")
        import pandas as pd
        dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=65)
        value = 100_000.0
        for i, d in enumerate(dates):
            if i > 30:
                value = value * (1 - 0.01)  # sustained -1%/day
            else:
                value = value * 1.002
            ledger.log_snapshot(str(d.date()), round(value, 2), 5000, {})
        _seed_live_decisions(ledger, n_days=65, correct_ratio=0.7)
        report = check_readiness(ledger, tickers=["TEST.NS"])
        assert report["gates"]["drawdown"]["passed"] is False


class TestIsReadyWrapper:
    def test_wrapper_returns_bool(self, ledger, tmp_path, monkeypatch):
        monkeypatch.setattr(readiness_check, "MODELS_DIR", str(tmp_path))
        model = os.path.join(str(tmp_path), "TEST_NS_models.pkl")
        with open(model, "w") as f:
            f.write("x")
        assert readiness_check.is_ready_for_live_trading() is False


class TestFormatReport:
    def test_renders_text(self):
        report = {
            "summary": "2/3 gates passed",
            "gates": {
                "paper_period": {"passed": True, "detail": "60/60 days"},
                "accuracy": {"passed": False, "detail": "no trades"},
            },
        }
        text = readiness_check.format_report(report)
        assert "[PASS] paper_period" in text
        assert "[FAIL] accuracy" in text

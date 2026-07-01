"""Tests for src/monitoring.py."""

import json
import os
from datetime import datetime, timedelta

import numpy as np
import pytest

from src.signals import monitoring
from src.signals.monitoring import (
    ModelMonitor,
    DriftAlert,
    MonitoringReport,
)


@pytest.fixture(autouse=True)
def temp_monitoring_dir(tmp_path, monkeypatch):
    """Redirect monitoring to temp dir."""
    monkeypatch.setattr(monitoring, "MONITORING_DIR", str(tmp_path / "monitoring"))
    os.makedirs(str(tmp_path / "monitoring"), exist_ok=True)
    yield


# ── DriftAlert ──

class TestDriftAlert:
    def test_creates_alert(self):
        alert = DriftAlert(
            metric="oos_accuracy", severity="warning",
            message="drift detected", current_value=0.48, threshold=0.50,
        )
        assert alert.metric == "oos_accuracy"
        assert alert.severity == "warning"
        assert alert.timestamp  # auto-set


# ── MonitoringReport ──

class TestMonitoringReport:
    def test_has_critical(self):
        report = MonitoringReport(ticker="T")
        report.alerts.append(DriftAlert("m", "critical", "bad", 0, 0))
        assert report.has_critical is True
        assert report.has_warnings is False

    def test_has_warnings(self):
        report = MonitoringReport(ticker="T")
        report.alerts.append(DriftAlert("m", "warning", "warn", 0, 0))
        assert report.has_warnings is True

    def test_no_alerts(self):
        report = MonitoringReport(ticker="T")
        assert report.has_critical is False
        assert report.has_warnings is False

    def test_to_dict(self):
        report = MonitoringReport(ticker="TEST.NS")
        d = report.to_dict()
        assert d["ticker"] == "TEST.NS"
        assert "alerts" in d
        assert "timestamp" in d


# ── ModelMonitor ──

class TestPerformanceDrift:
    def test_no_drift(self):
        monitor = ModelMonitor()
        alert = monitor.check_performance_drift("T", 0.53, 0.53)
        assert alert.severity == "info"

    def test_warning_level(self):
        monitor = ModelMonitor()
        alert = monitor.check_performance_drift("T", 0.50, 0.53)
        assert alert.severity == "warning"

    def test_critical_level(self):
        monitor = ModelMonitor()
        alert = monitor.check_performance_drift("T", 0.45, 0.53)
        assert alert.severity == "critical"

    def test_baseline_zero(self):
        monitor = ModelMonitor()
        alert = monitor.check_performance_drift("T", 0.50, 0.0)
        assert alert.severity == "info"

    def test_alert_persisted(self, tmp_path):
        monitor = ModelMonitor()
        monitor.check_performance_drift("TEST.NS", 0.48, 0.53)
        alerts_path = os.path.join(monitoring.MONITORING_DIR, "TEST_NS_alerts.json")
        assert os.path.exists(alerts_path)
        with open(alerts_path) as f:
            alerts = json.load(f)
        assert len(alerts) == 1
        assert alerts[0]["metric"] == "oos_accuracy"


class TestFeatureDrift:
    def test_no_drift(self):
        monitor = ModelMonitor()
        np.random.seed(42)
        baseline = np.random.randn(100, 5)
        current = np.random.randn(100, 5)
        alert = monitor.check_feature_drift("T", baseline, current)
        assert alert.severity in ("info", "warning")

    def test_detected_drift(self):
        monitor = ModelMonitor()
        np.random.seed(42)
        baseline = np.random.randn(100, 5)
        current = baseline + 5.0  # Huge shift
        alert = monitor.check_feature_drift("T", baseline, current)
        assert alert.severity in ("warning", "critical")

    def test_returns_p_value(self):
        monitor = ModelMonitor()
        np.random.seed(42)
        baseline = np.random.randn(50, 3)
        current = np.random.randn(50, 3)
        alert = monitor.check_feature_drift("T", baseline, current)
        assert alert.p_value is not None


class TestPredictionDrift:
    def test_no_drift(self):
        monitor = ModelMonitor()
        baseline = np.array([0.5, 0.5, 0.5, 0.5])
        current = np.array([0.51, 0.49, 0.50, 0.50])
        alert = monitor.check_prediction_drift("T", baseline, current)
        assert alert.severity == "info"

    def test_critical_drift(self):
        monitor = ModelMonitor()
        baseline = np.array([0.5, 0.5, 0.5])
        current = np.array([0.8, 0.8, 0.8])
        alert = monitor.check_prediction_drift("T", baseline, current)
        assert alert.severity == "critical"


class TestDataFreshness:
    def test_fresh(self):
        monitor = ModelMonitor()
        alert = monitor.check_data_freshness("T", datetime.now())
        assert alert.severity == "info"

    def test_warning(self):
        monitor = ModelMonitor()
        alert = monitor.check_data_freshness("T", datetime.now() - timedelta(days=5))
        assert alert.severity == "warning"

    def test_critical(self):
        monitor = ModelMonitor()
        alert = monitor.check_data_freshness("T", datetime.now() - timedelta(days=10))
        assert alert.severity == "critical"


class TestGenerateReport:
    def test_creates_report(self):
        monitor = ModelMonitor()
        report = monitor.generate_report("TEST.NS")
        assert report.ticker == "TEST.NS"
        report_path = os.path.join(monitoring.MONITORING_DIR, "TEST_NS_latest.json")
        assert os.path.exists(report_path)

    def test_includes_baseline(self):
        monitor = ModelMonitor()
        monitor.save_baseline("TEST.NS", {"accuracy": 0.53})
        report = monitor.generate_report("TEST.NS")
        assert "baseline" in report.metrics


class TestSaveBaseline:
    def test_saves_file(self):
        monitor = ModelMonitor()
        monitor.save_baseline("TEST.NS", {"accuracy": 0.53, "sharpe": 0.5})
        path = os.path.join(monitoring.MONITORING_DIR, "TEST_NS_baseline.json")
        assert os.path.exists(path)
        with open(path) as f:
            data = json.load(f)
        assert data["accuracy"] == 0.53


# ── Auto-retrain trigger ──

class TestRetainTrigger:
    """Critical drift on retrain-eligible metrics writes a trigger entry."""

    def test_critical_oos_accuracy_queues_trigger(self, tmp_path):
        monitor = ModelMonitor()
        # Patch _run_retrain_pipeline so the test doesn't actually train models
        monitor._run_retrain_pipeline = lambda ticker, path, entry: None

        monitor.check_performance_drift("TEST.NS", 0.45, 0.53)

        trigger_path = os.path.join(monitoring.MONITORING_DIR, "retrain_triggers.json")
        assert os.path.exists(trigger_path), "retrain_triggers.json should be created"
        with open(trigger_path) as f:
            triggers = json.load(f)
        assert len(triggers) >= 1
        assert triggers[-1]["ticker"] == "TEST.NS"
        assert triggers[-1]["metric"] == "oos_accuracy"
        assert triggers[-1]["status"] == "queued"

    def test_critical_feature_drift_queues_trigger(self):
        monitor = ModelMonitor()
        monitor._run_retrain_pipeline = lambda ticker, path, entry: None

        np.random.seed(0)
        baseline = np.random.randn(100, 5)
        current = baseline + 10.0  # guaranteed critical drift

        monitor.check_feature_drift("TEST.NS", baseline, current)

        trigger_path = os.path.join(monitoring.MONITORING_DIR, "retrain_triggers.json")
        assert os.path.exists(trigger_path)
        with open(trigger_path) as f:
            triggers = json.load(f)
        metrics = [t["metric"] for t in triggers]
        assert "feature_drift" in metrics

    def test_warning_does_not_trigger_retrain(self):
        monitor = ModelMonitor()
        monitor._run_retrain_pipeline = lambda ticker, path, entry: None

        # 5% drift → warning, not critical
        monitor.check_performance_drift("TEST.NS", 0.50, 0.53)

        trigger_path = os.path.join(monitoring.MONITORING_DIR, "retrain_triggers.json")
        # Either file doesn't exist, or has no entries for this ticker
        if os.path.exists(trigger_path):
            with open(trigger_path) as f:
                triggers = json.load(f)
            # No critical-level triggers should have been added
            critical = [t for t in triggers if t["ticker"] == "TEST.NS"]
            assert len(critical) == 0

    def test_prediction_drift_does_not_trigger_retrain(self):
        """prediction_drift is not in the retrain trigger set."""
        monitor = ModelMonitor()
        monitor._run_retrain_pipeline = lambda ticker, path, entry: None

        baseline = np.array([0.5, 0.5, 0.5])
        current = np.array([0.8, 0.8, 0.8])
        monitor.check_prediction_drift("TEST.NS", baseline, current)

        trigger_path = os.path.join(monitoring.MONITORING_DIR, "retrain_triggers.json")
        if os.path.exists(trigger_path):
            with open(trigger_path) as f:
                triggers = json.load(f)
            pred_triggers = [t for t in triggers if t["metric"] == "prediction_drift"]
            assert len(pred_triggers) == 0

    def test_triggers_capped_at_200(self):
        monitor = ModelMonitor()
        monitor._run_retrain_pipeline = lambda ticker, path, entry: None

        # Force 210 triggers by calling critical drift repeatedly
        for _ in range(210):
            monitor.check_performance_drift("TEST.NS", 0.40, 0.53)

        trigger_path = os.path.join(monitoring.MONITORING_DIR, "retrain_triggers.json")
        with open(trigger_path) as f:
            triggers = json.load(f)
        assert len(triggers) <= 200

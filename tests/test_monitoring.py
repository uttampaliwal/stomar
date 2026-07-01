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

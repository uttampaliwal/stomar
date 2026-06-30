"""Model monitoring and drift detection.

Detects:
- Performance degradation (OOS accuracy dropping)
- Feature drift (distribution shift in input features via KS test)
- Prediction drift (model output distribution shifting)
- Data freshness (stale data in pipeline)

Usage:
    from src.monitoring import ModelMonitor
    monitor = ModelMonitor()
    alert = monitor.check_performance_drift("RELIANCE.NS", 0.48, 0.53)
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

from src.constants import MONITORING_DIR


@dataclass
class DriftAlert:
    """A single drift detection alert."""
    metric: str
    severity: str  # "info", "warning", "critical"
    message: str
    current_value: float
    threshold: float
    p_value: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MonitoringReport:
    """Complete monitoring report for a ticker."""
    ticker: str
    alerts: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)

    @property
    def has_warnings(self) -> bool:
        return any(a.severity == "warning" for a in self.alerts)

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "has_critical": self.has_critical,
            "has_warnings": self.has_warnings,
            "n_alerts": len(self.alerts),
            "alerts": [
                {"metric": a.metric, "severity": a.severity, "message": a.message}
                for a in self.alerts
            ],
            "metrics": self.metrics,
        }


class ModelMonitor:
    """Monitor model health and data drift."""

    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        os.makedirs(MONITORING_DIR, exist_ok=True)

    def check_performance_drift(
        self, ticker: str, current_accuracy: float, baseline_accuracy: float
    ) -> DriftAlert:
        """Check if model accuracy has degraded."""
        if baseline_accuracy <= 0:
            drift_pct = 0.0
        else:
            drift_pct = (baseline_accuracy - current_accuracy) / baseline_accuracy

        if drift_pct > 0.10:
            severity = "critical"
        elif drift_pct > 0.05:
            severity = "warning"
        else:
            severity = "info"

        alert = DriftAlert(
            metric="oos_accuracy",
            severity=severity,
            message=f"OOS accuracy: {current_accuracy:.1%} (baseline: {baseline_accuracy:.1%}, drift: {drift_pct:.1%})",
            current_value=current_accuracy,
            threshold=baseline_accuracy * 0.90,
        )

        self._log_alert(ticker, alert)
        return alert

    def check_feature_drift(
        self, ticker: str, baseline_features: np.ndarray, current_features: np.ndarray
    ) -> DriftAlert:
        """Check for distribution shift in features using KS test."""
        n_features = min(baseline_features.shape[1], current_features.shape[1])
        p_values = []

        for i in range(n_features):
            _, p_value = stats.ks_2samp(baseline_features[:, i], current_features[:, i])
            p_values.append(p_value)

        mean_p = float(np.mean(p_values))
        n_drifted = sum(1 for p in p_values if p < 0.01)

        if n_drifted > n_features * 0.3:
            severity = "critical"
        elif n_drifted > n_features * 0.15:
            severity = "warning"
        else:
            severity = "info"

        alert = DriftAlert(
            metric="feature_drift",
            severity=severity,
            message=f"Feature drift: {n_drifted}/{n_features} features shifted (mean KS p={mean_p:.3f})",
            current_value=n_drifted / n_features if n_features > 0 else 0,
            threshold=0.15,
            p_value=mean_p,
        )

        self._log_alert(ticker, alert)
        return alert

    def check_prediction_drift(
        self, ticker: str, baseline_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> DriftAlert:
        """Check for distribution shift in model predictions."""
        baseline_mean = float(np.mean(baseline_predictions))
        current_mean = float(np.mean(current_predictions))
        shift = abs(current_mean - baseline_mean)

        if shift > 0.15:
            severity = "critical"
        elif shift > 0.10:
            severity = "warning"
        else:
            severity = "info"

        alert = DriftAlert(
            metric="prediction_drift",
            severity=severity,
            message=f"Prediction drift: mean shifted from {baseline_mean:.3f} to {current_mean:.3f} (delta={shift:.3f})",
            current_value=current_mean,
            threshold=baseline_mean + 0.10,
        )

        self._log_alert(ticker, alert)
        return alert

    def check_data_freshness(self, ticker: str, last_data_date: datetime) -> DriftAlert:
        """Check if data is fresh enough."""
        now = datetime.now()
        age_days = (now - last_data_date).days

        if age_days > 7:
            severity = "critical"
        elif age_days > 3:
            severity = "warning"
        else:
            severity = "info"

        alert = DriftAlert(
            metric="data_freshness",
            severity=severity,
            message=f"Data is {age_days} days old (last: {last_data_date.date()})",
            current_value=age_days,
            threshold=3,
        )

        self._log_alert(ticker, alert)
        return alert

    def generate_report(self, ticker: str) -> MonitoringReport:
        """Generate a monitoring report for a ticker.

        Reads baseline metrics and alert history, then writes the
        latest report to disk.
        """
        report = MonitoringReport(ticker=ticker)

        baseline_path = os.path.join(
            MONITORING_DIR, f"{ticker.replace('.', '_')}_baseline.json"
        )
        if os.path.exists(baseline_path):
            try:
                with open(baseline_path) as f:
                    report.metrics["baseline"] = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to read baseline for %s: %s", ticker, e)

        # Load recent alert history
        history_path = os.path.join(
            MONITORING_DIR, f"{ticker.replace('.', '_')}_alerts.json"
        )
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    alerts = json.load(f)
                # Include critical/warning alerts from last 24h
                from datetime import timedelta
                cutoff = (datetime.now() - timedelta(hours=24)).isoformat()
                recent = [a for a in alerts if a.get("timestamp", "") > cutoff
                          and a.get("severity") in ("warning", "critical")]
                report.metrics["recent_alerts"] = recent
            except (json.JSONDecodeError, OSError):
                pass

        report_path = os.path.join(
            MONITORING_DIR, f"{ticker.replace('.', '_')}_latest.json"
        )
        try:
            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)
        except OSError as e:
            logger.warning("Failed to write report for %s: %s", ticker, e)

        return report

    def save_baseline(self, ticker: str, metrics: dict):
        """Save baseline metrics for a ticker."""
        path = os.path.join(
            MONITORING_DIR, f"{ticker.replace('.', '_')}_baseline.json"
        )
        with open(path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved baseline metrics for {ticker}")

    def _log_alert(self, ticker: str, alert: DriftAlert):
        """Log an alert and persist it."""
        level = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}
        logger.log(level.get(alert.severity, logging.INFO), f"[{ticker}] {alert.message}")

        history_path = os.path.join(
            MONITORING_DIR, f"{ticker.replace('.', '_')}_alerts.json"
        )
        alerts = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                alerts = json.load(f)

        alerts.append({
            "metric": alert.metric,
            "severity": alert.severity,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "p_value": alert.p_value,
            "timestamp": alert.timestamp,
        })

        alerts = alerts[-100:]

        with open(history_path, "w") as f:
            json.dump(alerts, f, indent=2)

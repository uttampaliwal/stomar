"""Structured logging configuration for StoMar.

Provides:
- JSON-formatted logs for aggregation (ELK, Datadog, CloudWatch)
- Correlation IDs for distributed request tracing
- Thread-local context for pipeline/ticker association
- Prometheus-compatible metrics collection
"""

import json
import logging
import sys
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Optional

# ---------------------------------------------------------------------------
# Correlation / context storage (thread-local)
# ---------------------------------------------------------------------------

_context = threading.local()


def set_request_id(request_id: str):
    """Set the current request ID (from X-Request-ID header)."""
    _context.request_id = request_id


def get_request_id() -> str:
    """Get the current request ID, or empty string if not set."""
    return getattr(_context, "request_id", "")


def set_pipeline_context(pipeline: str = "", ticker: str = ""):
    """Set pipeline/ticker context for log lines during orchestrator runs."""
    _context.pipeline = pipeline
    _context.ticker = ticker


def clear_pipeline_context():
    """Clear pipeline context."""
    _context.pipeline = ""
    _context.ticker = ""


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------

class JSONFormatter(logging.Formatter):
    """JSON formatter for machine-parseable logs with correlation IDs."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "function": record.funcName,
            "message": record.getMessage(),
        }

        # Inject correlation context
        request_id = getattr(_context, "request_id", "")
        if request_id:
            log_entry["request_id"] = request_id

        pipeline = getattr(_context, "pipeline", "")
        if pipeline:
            log_entry["pipeline"] = pipeline
        ticker = getattr(_context, "ticker", "")
        if ticker:
            log_entry["ticker"] = ticker

        # Structured extra fields via `extra={"key": val}` in log calls
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data

        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_entry, default=str)


class HumanFormatter(logging.Formatter):
    """Human-readable formatter for terminal output."""

    def format(self, record):
        ts = datetime.now().strftime("%H:%M:%S")
        request_id = getattr(_context, "request_id", "")
        rid = f" [{request_id[:8]}]" if request_id else ""
        return f"{ts} {record.levelname:8s} [{record.module}]{rid} {record.getMessage()}"


# ---------------------------------------------------------------------------
# Prometheus-style metrics (in-process, no dependency)
# ---------------------------------------------------------------------------

class MetricsCollector:
    """Lightweight in-process metrics for Prometheus export.

    Tracks:
    - http_requests_total{method, path, status}
    - http_request_duration_seconds{method, path}  (histogram buckets)
    - pipeline_runs_total{status}
    - pipeline_stage_duration_seconds{stage}
    - cache_hits_total / cache_misses_total
    """

    def __init__(self):
        self._counters: Counter = Counter()
        self._histograms: dict[str, list[float]] = {}
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()

    def inc(self, name: str, labels: dict[str, str] | None = None, value: float = 1):
        key = self._key(name, labels)
        with self._lock:
            self._counters[key] += value

    def observe(self, name: str, value: float, labels: dict[str, str] | None = None):
        key = self._key(name, labels)
        with self._lock:
            if key not in self._histograms:
                self._histograms[key] = []
            self._histograms[key].append(value)

    def gauge(self, name: str, value: float, labels: dict[str, str] | None = None):
        key = self._key(name, labels)
        with self._lock:
            self._gauges[key] = value

    def _key(self, name: str, labels: dict | None) -> str:
        if labels:
            label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
            return f"{name}{{{label_str}}}"
        return name

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text exposition format."""
        lines = []
        with self._lock:
            for key, val in sorted(self._counters.items()):
                lines.append(f"# TYPE {key.split('{')[0]} counter")
                lines.append(f"{key} {val}")

            for key, values in sorted(self._histograms.items()):
                base = key.split("{")[0]
                lines.append(f"# TYPE {base} histogram")
                sorted_vals = sorted(values)
                n = len(sorted_vals)
                for bucket in [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]:
                    count = sum(1 for v in sorted_vals if v <= bucket)
                    label = f'{key.split("{")[0]}{{le="{bucket}"' + ('",' + key.split("{")[1] if '{' in key else '"')
                    lines.append(f"{label} {count}")
                label_all = f'{base}{{le="+Inf"' + (',' + key.split("{")[1] if '{' in key else '"')
                lines.append(f"{label_all} {n}")
                lines.append(f"{base}_sum{('{' + key.split('{')[1] if '{' in key else '')} {sum(values)}")
                lines.append(f"{base}_count{('{' + key.split('{')[1] if '{' in key else '')} {n}")

            for key, val in sorted(self._gauges.items()):
                lines.append(f"# TYPE {key.split('{')[0]} gauge")
                lines.append(f"{key} {val}")

        return "\n".join(lines) + "\n"


# Global metrics singleton
metrics = MetricsCollector()


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO", json_output: bool = False) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: If True, use JSON format; otherwise human-readable
    """
    logger = logging.getLogger("stomar")
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter() if json_output else HumanFormatter())
    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """Get a child logger for a specific module."""
    return logging.getLogger(f"stomar.{name}")

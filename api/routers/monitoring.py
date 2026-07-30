"""System monitoring endpoint."""

import json
import os
from datetime import datetime, timedelta

from fastapi import APIRouter

from src.core.constants import MONITORING_DIR
from src.core.logging_config import metrics
from src.core.pipeline_state import PipelineCheckpoint, checkpoint_path
from src.data.data_fetcher import NSE_STOCKS
from src.models.model import models_exist
from src.signals.monitoring import ModelMonitor

router = APIRouter()

_PIPELINE_FAILURES_PATH = os.path.join(MONITORING_DIR, "pipeline_failures.json")
_RETRAIN_TRIGGERS_PATH = os.path.join(MONITORING_DIR, "retrain_triggers.json")


@router.get("/")
def get_monitoring():
    try:
        monitor = ModelMonitor()
        alerts = {"critical": [], "warning": [], "info": []}
        models_trained = []
        models_missing = []

        for ticker in NSE_STOCKS:
            if models_exist(ticker):
                models_trained.append(ticker)
            else:
                models_missing.append(ticker)

        if models_missing:
            alerts["warning"].append({
                "severity": "warning",
                "message": f"{len(models_missing)} models not trained: {', '.join(models_missing[:5])}",
            })

        if not models_trained:
            alerts["critical"].append({
                "severity": "critical",
                "message": "No models trained. Run pipeline first.",
            })

        alert_history = []
        history_path = os.path.join(MONITORING_DIR, "alert_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path) as f:
                    alert_history = json.load(f)[-10:]
            except (json.JSONDecodeError, OSError):
                alert_history = []

        # Surface recent pipeline failures
        recent_failures = _get_recent_failures(hours=24)
        if recent_failures:
            alerts["critical"].append({
                "severity": "critical",
                "message": f"{len(recent_failures)} pipeline failure(s) in last 24h — check /api/monitoring/failures",
            })

        return {
            "models_trained": models_trained,
            "models_missing": models_missing,
            "alerts": alerts,
            "alert_history": alert_history,
            "pipeline_failures_last_24h": len(recent_failures),
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/failures")
def get_pipeline_failures(hours: int = 48):
    """Return pipeline failures from the last N hours (default 48).

    The frontend uses this to display a persistent red banner when the
    pipeline has failed since the last successful run.
    """
    failures = _get_recent_failures(hours=hours)
    return {
        "count": len(failures),
        "failures": failures,
        "hours_lookback": hours,
    }


@router.get("/metrics")
def get_pipeline_metrics():
    """Return pipeline execution metrics for observability.

    Aggregates failure counts by stage, recent run durations,
    and success/failure rates.
    """
    failures = _get_recent_failures(hours=168)  # 7 days

    # Aggregate by stage
    stage_counts = {}
    for f in failures:
        stage = f.get("stage", "unknown")
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    # Compute success rate (from checkpoint history)
    recent_checkpoints = _get_recent_checkpoint_summaries()

    return {
        "failures_by_stage": stage_counts,
        "total_failures_7d": len(failures),
        "recent_runs": recent_checkpoints,
        "cache_stats": {
            "hits": int(metrics._counters.get("cache_hits_total", 0)),
            "misses": int(metrics._counters.get("cache_misses_total", 0)),
        },
    }


@router.get("/retrain-triggers")
def get_retrain_triggers():
    """Return the history of automatic retraining triggers."""
    if not os.path.exists(_RETRAIN_TRIGGERS_PATH):
        return {"count": 0, "triggers": []}
    try:
        with open(_RETRAIN_TRIGGERS_PATH) as f:
            triggers = json.load(f)
        return {"count": len(triggers), "triggers": triggers[-50:]}
    except (json.JSONDecodeError, OSError) as e:
        return {"error": str(e), "count": 0, "triggers": []}


@router.get("/checkpoint")
def get_pipeline_checkpoint():
    """Return the current pipeline checkpoint status.

    Shows whether the auto-pipeline or any retraining pipeline has an
    interrupted run that can be resumed on next startup.
    """
    # Auto-pipeline checkpoint
    auto_ckpt = PipelineCheckpoint.load(pipeline="auto")
    auto_status = auto_ckpt.get_summary() if auto_ckpt else None

    # Per-ticker retraining checkpoints
    retrain_checkpoints = []
    for ticker in NSE_STOCKS:
        ckpt = PipelineCheckpoint.load(pipeline="retrain", ticker=ticker)
        if ckpt and not ckpt.is_complete():
            retrain_checkpoints.append(ckpt.get_summary())

    return {
        "auto_pipeline": auto_status,
        "retrain_pipelines": retrain_checkpoints,
        "has_pending_checkpoints": (
            (auto_status is not None and auto_status.get("can_resume", False))
            or len(retrain_checkpoints) > 0
        ),
    }


def _get_recent_failures(hours: int = 24) -> list[dict]:
    """Load pipeline_failures.json and return entries within the last N hours."""
    if not os.path.exists(_PIPELINE_FAILURES_PATH):
        return []
    try:
        with open(_PIPELINE_FAILURES_PATH) as f:
            failures = json.load(f)
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        return [f for f in failures if f.get("timestamp", "") >= cutoff]
    except (json.JSONDecodeError, OSError):
        return []


def _get_recent_checkpoint_summaries() -> list[dict]:
    """Load recent checkpoint data for metrics."""
    ckpt = PipelineCheckpoint.load(pipeline="auto")
    if not ckpt:
        return []
    summary = ckpt.get_summary()
    return [{
        "pipeline": summary.get("pipeline"),
        "current_stage": summary.get("current_stage"),
        "is_complete": summary.get("is_complete"),
        "has_failed": summary.get("has_failed"),
        "updated_at": summary.get("updated_at"),
    }]

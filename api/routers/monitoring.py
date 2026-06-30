"""System monitoring endpoint."""

import os
import json
from fastapi import APIRouter
from src.monitoring import ModelMonitor, MONITORING_DIR
from src.data_fetcher import NSE_STOCKS, fetch_stock_data
from src.model import models_exist

router = APIRouter()


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
            with open(history_path) as f:
                alert_history = json.load(f)[-10:]

        return {
            "models_trained": models_trained,
            "models_missing": models_missing,
            "alerts": alerts,
            "alert_history": alert_history,
        }
    except Exception as e:
        return {"error": str(e)}

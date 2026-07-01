from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.signals.orchestrator import DailyOrchestrator
from src.models.meta_controller import MetaController


class DailyAutomationRunner:
    """One-command automation wrapper for daily signal generation and logging."""

    def __init__(self, tickers: list[str] | None = None, db_path: str | None = None):
        self.tickers = tickers or list(NSE_STOCKS)
        self.db_path = db_path

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        ledger = Ledger(self.db_path)
        meta_controller = MetaController()
        model_path = Path(__file__).resolve().parent.parent.parent / "models" / "meta_controller.pkl"
        if model_path.exists():
            import joblib
            try:
                meta_controller = joblib.load(model_path)
            except Exception:
                meta_controller = MetaController()

        orchestrator = DailyOrchestrator(tickers=self.tickers, ledger=ledger, meta_controller=meta_controller)
        summary = orchestrator.run(date=datetime.now().strftime("%Y-%m-%d"), dry_run=dry_run)
        ledger.close()
        return summary

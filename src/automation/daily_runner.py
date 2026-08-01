from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.signals.orchestrator import DailyOrchestrator
from src.models.meta_controller import MetaController
from src.core.constants import META_CONTROLLER_PATH

logger = logging.getLogger(__name__)


class DailyAutomationRunner:
    """One-command automation wrapper for daily signal generation and logging."""

    def __init__(self, tickers: list[str] | None = None, db_path: str | None = None):
        self.tickers = tickers or list(NSE_STOCKS)
        self.db_path = db_path

    def run(self, dry_run: bool = False) -> dict[str, Any]:
        from src.core.trading_mode import get_trading_mode, mode_banner
        mode = get_trading_mode()
        logger.info("daily runner mode: %s", mode.value)
        if mode.value == "live":
            logger.warning(mode_banner(mode))

        ledger = Ledger(self.db_path)
        meta_controller = MetaController()
        if os.path.exists(META_CONTROLLER_PATH):
            meta_controller.load()  # manifest-verified; refuses unverified artifacts

        orchestrator = DailyOrchestrator(tickers=self.tickers, ledger=ledger, meta_controller=meta_controller)
        summary = orchestrator.run(date=datetime.now().strftime("%Y-%m-%d"), dry_run=dry_run)
        ledger.close()
        return summary

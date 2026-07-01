"""Auto-pipeline: startup automation for StoMar.

Detects missed trading days, backfills them, trains models,
runs the daily orchestrator, and auto-executes paper trades.

Designed to run:
  1. As a standalone script: python auto_pipeline.py
  2. Via Windows Task Scheduler (auto-installed on first run)

Usage:
    from auto_pipeline import AutoPipeline
    pipeline = AutoPipeline()
    pipeline.run()  # detects gaps, backfills, runs daily, paper trades
"""

import logging
import os
import sys
import time
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.core.constants import (
    DATA_DIR, MODELS_DIR, LEDGER_DB, META_CONTROLLER_PATH,
)
from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.core.logging_config import get_logger

logger = get_logger("auto_pipeline")


def _is_business_day(dt) -> bool:
    """Check if a date is a business day (Mon-Fri)."""
    return dt.weekday() < 5


def _get_trading_days_since(start_date: str, end_date: str) -> list[str]:
    """Get business days between start and end dates (inclusive)."""
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    days = []
    current = start
    while current <= end:
        if _is_business_day(current):
            days.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return days


class AutoPipeline:
    """Fully automated pipeline: detect gaps -> backfill -> train -> daily run -> paper trade."""

    def __init__(self, tickers: list[str] = None):
        self.tickers = tickers or NSE_STOCKS
        self.ledger = None
        self._running = False
        self._last_run_date = None
        self._status = "idle"
        self._log_lines = []

    @property
    def status(self) -> str:
        return self._status

    @property
    def last_run_date(self) -> str:
        return self._last_run_date

    @property
    def log(self) -> list[str]:
        return self._log_lines[-50:]  # Keep last 50 lines

    def _log(self, msg: str):
        """Log to both logger and internal buffer."""
        ts = datetime.now().strftime("%H:%M:%S")
        line = f"[{ts}] {msg}"
        self._log_lines.append(line)
        if len(self._log_lines) > 200:
            self._log_lines = self._log_lines[-100:]
        logger.info(msg)

    def run(self, force: bool = False) -> dict:
        """Run the full auto-pipeline.

        Args:
            force: If True, skip "already ran today" check

        Returns:
            Summary dict with backfill, daily, paper trade results
        """
        if self._running:
            self._log("Pipeline already running, skipping")
            return {"status": "already_running"}

        self._running = True
        self._status = "running"
        result = {
            "status": "success",
            "backfill": None,
            "daily": None,
            "paper_trade": None,
            "meta_controller": None,
            "scheduler": None,
            "errors": [],
        }

        try:
            self.ledger = Ledger()
            today = datetime.now().strftime("%Y-%m-%d")
            last_date = self.ledger.get_last_decision_date()

            self._log(f"Auto-pipeline start | today={today} | last_decision={last_date}")

            # Step 0: Auto-install scheduler if not present
            scheduler_result = self._auto_install_scheduler()
            result["scheduler"] = scheduler_result

            # Check if we already ran today
            if not force and last_date == today:
                existing = self.ledger.get_dates_with_decisions()
                if today in existing:
                    self._log("Already ran today, skipping (use force=True to override)")
                    self._status = "completed_today"
                    result["status"] = "already_completed_today"
                    return result

            # Step 1: Detect and backfill missed days
            backfill_result = self._backfill_missed_days(today, last_date)
            result["backfill"] = backfill_result

            # Step 2: Train meta-controller if needed
            mc_result = self._ensure_meta_controller()
            result["meta_controller"] = mc_result

            # Step 3: Run daily orchestrator
            daily_result = self._run_daily(today)
            result["daily"] = daily_result

            # Step 4: Auto-execute paper trades
            paper_result = self._run_paper_trades()
            result["paper_trade"] = paper_result

            self._last_run_date = today
            self._status = "completed"
            self._log(f"Auto-pipeline complete | backfill={backfill_result.get('days_backfilled', 0)} days | daily={daily_result.get('decisions', 0)} decisions | paper={paper_result.get('trades', 0)} trades")

        except Exception as e:
            self._log(f"Pipeline error: {e}")
            self._status = "error"
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.exception("Auto-pipeline failed")
        finally:
            self._running = False
            if self.ledger:
                self.ledger.close()

        return result

    def _backfill_missed_days(self, today: str, last_date: str | None) -> dict:
        """Detect and backfill missed trading days."""
        result = {"days_backfilled": 0, "tickers_processed": 0}

        if last_date is None:
            # No history at all - do a full backfill
            self._log("No history found, running full 1-year backfill")
            return self._run_backfill(252)

        # Calculate missed trading days
        missed_days = _get_trading_days_since(last_date, today)
        # Remove the last_date itself (already processed) and today (will be processed by daily)
        missed_days = [d for d in missed_days if d != last_date]

        # Remove today - the daily orchestrator handles today
        if today in missed_days:
            missed_days.remove(today)

        if not missed_days:
            self._log("No missed days to backfill")
            return result

        self._log(f"Detected {len(missed_days)} missed trading days: {missed_days[0]} to {missed_days[-1]}")

        # Backfill is idempotent (skips existing decisions)
        days_to_backfill = len(missed_days) + 5  # Small buffer for warmup
        return self._run_backfill(min(days_to_backfill, 30))  # Cap at 30 days for performance

    def _run_backfill(self, lookback_days: int) -> dict:
        """Run the historical backfill."""
        self._log(f"Running backfill ({lookback_days} days)")
        try:
            from src.core.backfill import HistoricalBackfill
            backfill = HistoricalBackfill(self.ledger)
            summary = backfill.run(tickers=self.tickers, lookback_days=lookback_days)
            days_backfilled = summary.get("total_decisions", 0)
            self._log(f"Backfill complete: {days_backfilled} decisions")
            return {"days_backfilled": days_backfilled, "tickers_processed": len(self.tickers)}
        except Exception as e:
            self._log(f"Backfill error: {e}")
            return {"error": str(e), "days_backfilled": 0}

    def _ensure_meta_controller(self) -> dict:
        """Load or train meta-controller."""
        result = {"status": "not_needed"}
        try:
            import joblib
            from src.models.meta_controller import MetaController

            if os.path.exists(META_CONTROLLER_PATH):
                mc = joblib.load(META_CONTROLLER_PATH)
                self._log("Loaded pre-trained meta-controller")
                result["status"] = "loaded"
                return result

            # Train fresh
            self._log("No meta-controller found, training...")
            mc = MetaController()
            train_result = mc.train(self.ledger)

            if train_result["status"] == "trained":
                os.makedirs(MODELS_DIR, exist_ok=True)
                joblib.dump(mc, META_CONTROLLER_PATH)
                self._log(f"Meta-controller trained: accuracy={train_result['accuracy']:.1%}")
                result["status"] = "trained"
                result["accuracy"] = train_result["accuracy"]
            else:
                self._log(f"Meta-controller training skipped: {train_result.get('n_samples', 0)} samples (need 100+)")
                result["status"] = "insufficient_data"
        except Exception as e:
            self._log(f"Meta-controller error: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def _run_daily(self, today: str) -> dict:
        """Run the daily orchestrator."""
        self._log(f"Running daily orchestrator for {today}")
        try:
            import joblib
            from src.signals.orchestrator import DailyOrchestrator
            from src.models.meta_controller import MetaController

            meta_controller = None
            if os.path.exists(META_CONTROLLER_PATH):
                meta_controller = joblib.load(META_CONTROLLER_PATH)

            orchestrator = DailyOrchestrator(
                tickers=self.tickers,
                ledger=self.ledger,
                meta_controller=meta_controller,
            )

            summary = orchestrator.run(date=today)
            n_decisions = len(summary.get("decisions", []))
            n_errors = len(summary.get("errors", []))
            self._log(f"Daily complete: {n_decisions} decisions, {n_errors} errors")
            return {"decisions": n_decisions, "errors": n_errors}
        except Exception as e:
            self._log(f"Daily orchestrator error: {e}")
            return {"error": str(e), "decisions": 0}

    def _run_paper_trades(self) -> dict:
        """Auto-execute paper trades based on today's signals."""
        result = {"trades": 0}
        try:
            from src.core.constants import PAPER_STATE_PATH
            if not os.path.exists(PAPER_STATE_PATH):
                self._log("No paper state found, skipping paper trades")
                return result

            # Get today's decisions from the ledger
            today = datetime.now().strftime("%Y-%m-%d")
            decisions = self.ledger.get_decisions(start_date=today, end_date=today)
            if not decisions:
                self._log("No decisions today, skipping paper trades")
                return result

            from run_daily import run_paper_trades
            self._log(f"Executing paper trades for {len(decisions)} decisions...")
            summary = run_paper_trades(decisions, self.ledger)
            result["trades"] = summary.get("total_trades", 0)
            self._log(f"Paper trades executed: {result['trades']} trades")
        except Exception as e:
            self._log(f"Paper trade error: {e}")
            result["error"] = str(e)
        return result

    def _auto_install_scheduler(self) -> dict:
        """Auto-install Windows Task Scheduler if not already present."""
        result = {"status": "skipped"}
        try:
            import subprocess
            TASK_NAME = "StoMar_Daily_Signal"
            cmd = ["schtasks", "/query", "/tn", TASK_NAME, "/fo", "LIST"]
            check = subprocess.run(cmd, capture_output=True, text=True)

            if check.returncode == 0:
                self._log("Scheduler already installed")
                result["status"] = "already_installed"
                return result

            # Not installed - install it
            self._log("Installing Windows Task Scheduler (daily 4 PM)...")
            from schedule_pipeline import install_task
            ret = install_task("16:00")
            result["status"] = "installed" if ret == 0 else "failed"
            result["returncode"] = ret
        except Exception as e:
            self._log(f"Scheduler auto-install failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def install_scheduler(self) -> dict:
        """Install Windows Task Scheduler for daily 4 PM run."""
        try:
            from schedule_pipeline import install_task
            result = install_task()
            self._log(f"Scheduler installed: {result}")
            return result
        except Exception as e:
            self._log(f"Scheduler install failed: {e}")
            return {"status": "error", "error": str(e)}

    def warm_sentiment_cache(self, tickers: list[str] = None) -> dict:
        """Pre-warm sentiment cache for given tickers."""
        tickers = tickers or self.tickers[:5]  # Top 5 only for speed
        self._log(f"Warming sentiment cache for {len(tickers)} tickers...")
        warmed = 0
        errors = 0
        for ticker in tickers:
            try:
                from src.signals.sentiment import get_stock_sentiment
                get_stock_sentiment(ticker)
                warmed += 1
            except Exception:
                errors += 1
        self._log(f"Sentiment warm-up: {warmed} cached, {errors} errors")
        return {"warmed": warmed, "errors": errors}


def main():
    """CLI entry point for auto-pipeline."""
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="StoMar Auto-Pipeline")
    parser.add_argument("--force", action="store_true",
                        help="Force run even if already ran today")
    parser.add_argument("--install-scheduler", action="store_true",
                        help="Install Windows Task Scheduler")
    parser.add_argument("--warm-sentiment", action="store_true",
                        help="Pre-warm sentiment cache only")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="Tickers to process (default: all NSE stocks)")
    args = parser.parse_args()

    pipeline = AutoPipeline(tickers=args.ticker)

    if args.install_scheduler:
        result = pipeline.install_scheduler()
        print(f"Scheduler: {result}")
        return

    if args.warm_sentiment:
        result = pipeline.warm_sentiment_cache()
        print(f"Sentiment warm-up: {result}")
        return

    result = pipeline.run(force=args.force)
    print(f"\nPipeline result: {result['status']}")
    if result.get("backfill"):
        print(f"  Backfill: {result['backfill'].get('days_backfilled', 0)} days")
    if result.get("daily"):
        print(f"  Daily: {result['daily'].get('decisions', 0)} decisions")
    if result.get("paper_trade"):
        print(f"  Paper trades: {result['paper_trade'].get('trades', 0)}")
    if result.get("errors"):
        print(f"  Errors: {result['errors']}")


if __name__ == "__main__":
    main()

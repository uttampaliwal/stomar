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

import json
import logging
import os
import signal
import sys
import tempfile
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from src.core.constants import (
    MODELS_DIR, META_CONTROLLER_PATH, MONITORING_DIR,
)
from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.core.logging_config import get_logger
from src.core.pipeline_state import PipelineCheckpoint, StageInfo, StageStatus

logger = get_logger("auto_pipeline")

# Path for pipeline failure records surfaced on the Monitoring page
PIPELINE_FAILURES_PATH = os.path.join(MONITORING_DIR, "pipeline_failures.json")


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
        self._shutdown_requested = False
        self._ckpt = None

    def _setup_signal_handlers(self):
        """Register signal handlers for graceful shutdown."""
        def _handle_shutdown(signum, frame):
            sig_name = signal.Signals(signum).name
            logger.warning("Received %s — initiating graceful shutdown", sig_name)
            self._log(f"Shutdown signal received ({sig_name})")
            self._shutdown_requested = True
            self._finalize_checkpoint_on_shutdown()

        signal.signal(signal.SIGTERM, _handle_shutdown)
        signal.signal(signal.SIGINT, _handle_shutdown)

    def _finalize_checkpoint_on_shutdown(self):
        """Save checkpoint with any RUNNING stages marked as failed."""
        if self._ckpt is None:
            return
        try:
            for stage_name, info in self._ckpt.stages.items():
                if info.status == StageStatus.RUNNING:
                    self._ckpt.fail(stage_name, "interrupted by signal")
            self._ckpt.save()
            logger.info("Checkpoint finalized on shutdown")
        except Exception as e:
            logger.error("Failed to finalize checkpoint on shutdown: %s", e)

    def _check_shutdown(self) -> bool:
        """Return True if shutdown was requested, logging the message."""
        if self._shutdown_requested:
            self._log("Shutdown requested — stopping pipeline")
            return True
        return False

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

    # ── Failure notification ─────────────────────────────────────────────────

    def _record_failure(self, stage: str, error: str):
        """Persist a pipeline failure to MONITORING_DIR/pipeline_failures.json.

        The Monitoring page polls this file and displays a banner when failures
        exist from the last 24 hours.
        """
        os.makedirs(MONITORING_DIR, exist_ok=True)
        failures = []
        if os.path.exists(PIPELINE_FAILURES_PATH):
            try:
                with open(PIPELINE_FAILURES_PATH) as f:
                    failures = json.load(f)
            except (json.JSONDecodeError, OSError):
                failures = []

        failures.append({
            "stage": stage,
            "error": str(error)[:500],
            "timestamp": datetime.now().isoformat(),
        })
        failures = failures[-100:]  # keep last 100 entries

        # Atomic write
        dir_name = os.path.dirname(PIPELINE_FAILURES_PATH) or "."
        fd, tmp = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(failures, f, indent=2)
            os.replace(tmp, PIPELINE_FAILURES_PATH)
        except Exception:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

        logger.critical("Pipeline failure recorded | stage=%s | error=%s", stage, error)

    def run(self, force: bool = False) -> dict:
        """Run the full auto-pipeline with checkpoint-based resume.

        On startup, reads data/pipeline_checkpoint.json. If a valid
        checkpoint exists from a previous interrupted run, resumes from
        the last incomplete stage. Otherwise starts fresh.

        Args:
            force: If True, skip "already ran today" check

        Returns:
            Summary dict with backfill, daily, paper trade results
        """
        if self._running:
            self._log("Pipeline already running, skipping")
            return {"status": "already_running"}

        self._setup_signal_handlers()
        self._running = True
        self._status = "running"
        result = {
            "status": "success",
            "backfill": None,
            "daily": None,
            "paper_trade": None,
            "meta_controller": None,
            "scheduler": None,
            "checkpoint": None,
            "errors": [],
        }

        # ── Load or create checkpoint ──────────────────────────────────────
        ckpt = PipelineCheckpoint.load()
        if ckpt and ckpt.can_resume():
            self._log(
                f"Resuming from checkpoint | pipeline={ckpt.pipeline} "
                f"| stage={ckpt.resume_from()} | created={ckpt.created_at}"
            )
            result["checkpoint"] = {"resumed_from": ckpt.resume_from(), "created_at": ckpt.created_at}
        else:
            ckpt = PipelineCheckpoint(pipeline="auto")
            PipelineCheckpoint.clear()
        self._ckpt = ckpt

        try:
            self.ledger = Ledger()
            today = datetime.now().strftime("%Y-%m-%d")
            last_date = self.ledger.get_last_decision_date()

            self._log(f"Auto-pipeline start | today={today} | last_decision={last_date}")

            # Step 0: Auto-install scheduler if not present
            scheduler_result = self._auto_install_scheduler()
            result["scheduler"] = scheduler_result

            # Check if we already ran today (skip if resuming from checkpoint)
            if not force and last_date == today and not ckpt.can_resume():
                existing = self.ledger.get_dates_with_decisions()
                if today in existing:
                    self._log("Already ran today, skipping (use force=True to override)")
                    self._status = "completed_today"
                    result["status"] = "already_completed_today"
                    return result

            # Step 1: Detect and backfill missed days
            if self._check_shutdown():
                result["status"] = "interrupted"
                return result
            if ckpt.stage_status("FETCH") not in ("completed", "skipped"):
                ckpt.advance("FETCH")
                backfill_result = self._backfill_missed_days(today, last_date)
                result["backfill"] = backfill_result
                if backfill_result.get("error"):
                    ckpt.fail("FETCH", backfill_result["error"])
                    ckpt.save()
                    raise RuntimeError(f"Backfill failed: {backfill_result['error']}")
                elif backfill_result.get("days_backfilled", 0) == 0:
                    ckpt.skip("FETCH", "no missed days")
                else:
                    ckpt.complete("FETCH", backfill_result)
                ckpt.save()
            else:
                self._log("FETCH already completed, skipping")
                result["backfill"] = ckpt.stages.get("FETCH", StageInfo()).result or {"days_backfilled": 0}

            # Step 2: Train meta-controller if needed
            if self._check_shutdown():
                result["status"] = "interrupted"
                return result
            if ckpt.stage_status("TRAIN") not in ("completed", "skipped"):
                ckpt.advance("TRAIN")
                mc_result = self._ensure_meta_controller()
                result["meta_controller"] = mc_result
                if mc_result.get("status") == "error":
                    ckpt.fail("TRAIN", mc_result.get("error", "unknown"))
                    ckpt.save()
                    raise RuntimeError(f"Meta-controller failed: {mc_result.get('error')}")
                elif mc_result.get("status") in ("loaded", "insufficient_data"):
                    ckpt.skip("TRAIN", mc_result.get("status"))
                else:
                    ckpt.complete("TRAIN", mc_result)
                ckpt.save()
            else:
                self._log("TRAIN already completed, skipping")
                result["meta_controller"] = {"status": "skipped"}

            # Step 3: Run daily orchestrator
            if self._check_shutdown():
                result["status"] = "interrupted"
                return result
            if ckpt.stage_status("DAILY") not in ("completed", "skipped"):
                ckpt.advance("DAILY")
                daily_result = self._run_daily(today)
                result["daily"] = daily_result
                if daily_result.get("error"):
                    ckpt.fail("DAILY", daily_result["error"])
                    ckpt.save()
                    raise RuntimeError(f"Daily orchestrator failed: {daily_result['error']}")
                ckpt.complete("DAILY", daily_result)
                ckpt.save()
            else:
                self._log("DAILY already completed, skipping")
                result["daily"] = {"decisions": 0, "skipped": True}

            # Step 4: Auto-execute paper trades
            if self._check_shutdown():
                result["status"] = "interrupted"
                return result
            if ckpt.stage_status("PAPER") not in ("completed", "skipped"):
                ckpt.advance("PAPER")
                paper_result = self._run_paper_trades()
                result["paper_trade"] = paper_result
                if paper_result.get("error"):
                    ckpt.fail("PAPER", paper_result["error"])
                    ckpt.save()
                    raise RuntimeError(f"Paper trades failed: {paper_result['error']}")
                ckpt.complete("PAPER", paper_result)
                ckpt.save()
            else:
                self._log("PAPER already completed, skipping")
                result["paper_trade"] = {"trades": 0, "skipped": True}

            # Mark pipeline done
            ckpt.advance("DONE")
            ckpt.complete("DONE")
            ckpt.save()

            self._last_run_date = today
            self._status = "completed"
            self._log(
                f"Auto-pipeline complete | "
                f"backfill={result.get('backfill', {}).get('days_backfilled', 0)} days | "
                f"daily={result.get('daily', {}).get('decisions', 0)} decisions | "
                f"paper={result.get('paper_trade', {}).get('trades', 0)} trades"
            )

            # Clear checkpoint on successful completion
            PipelineCheckpoint.clear()

        except Exception as e:
            self._log(f"Pipeline error: {e}")
            self._status = "error"
            result["status"] = "error"
            result["errors"].append(str(e))
            logger.exception("Auto-pipeline failed")
            self._record_failure(stage="pipeline", error=str(e))
            # Save checkpoint on failure so we can resume later
            try:
                ckpt.save()
            except Exception:
                pass
        finally:
            self._running = False
            self._ckpt = None
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
            from src.models.meta_controller import MetaController

            if os.path.exists(META_CONTROLLER_PATH):
                mc = MetaController()
                if mc.load():
                    self._log("Loaded pre-trained meta-controller")
                    result["status"] = "loaded"
                    return result
                self._log("Meta-controller artifact failed verification — retraining")

            # Train fresh
            self._log("No meta-controller found, training...")
            mc = MetaController()
            train_result = mc.train(self.ledger)

            if train_result["status"] == "trained":
                os.makedirs(MODELS_DIR, exist_ok=True)
                mc.save()
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
        """Run the daily orchestrator.

        Resets the RiskController's daily P&L counter at the start of each
        new trading day (P0.4 fix) so the 2% daily loss limit is evaluated
        on today's activity only, not accumulated across multiple days.
        """
        self._log(f"Running daily orchestrator for {today}")
        try:
            from src.signals.orchestrator import DailyOrchestrator
            from src.core.trading_mode import get_trading_mode, mode_banner

            mode = get_trading_mode()
            self._log(f"Trading mode: {mode.value}")
            if mode.value == "live":
                self._log(mode_banner(mode))
                from src.core.trading_mode import require_live_allowed
                require_live_allowed()

            meta_controller = None
            if os.path.exists(META_CONTROLLER_PATH):
                from src.models.meta_controller import MetaController
                meta_controller = MetaController()
                if not meta_controller.load():
                    meta_controller = None

            # ── P0.4: reset daily P&L on the paper trader ─────────────────
            paper_trader = None
            from src.core.constants import PAPER_STATE_PATH
            from filelock import FileLock
            state_lock = FileLock(os.path.join(os.path.dirname(PAPER_STATE_PATH), "paper_state.json.lock"), timeout=30)
            try:
                from src.trading.paper_trader import PaperTrader
                state_lock.acquire()
                paper_trader = PaperTrader(initial_capital=200_000)
                paper_trader.load_state()
                paper_trader.risk_controller.reset_daily()
                self._log("Daily P&L counter reset on paper trader")
            except Exception as e:
                self._log(f"Paper trader init warning: {e}")
            # ──────────────────────────────────────────────────────────────

            orchestrator = DailyOrchestrator(
                tickers=self.tickers,
                ledger=self.ledger,
                meta_controller=meta_controller,
                paper_trader=paper_trader,
            )

            summary = orchestrator.run(date=today)
            n_decisions = len(summary.get("decisions", []))
            n_errors = len(summary.get("errors", []))

            # Persist updated paper trader state after the daily run
            if paper_trader is not None:
                try:
                    paper_trader.save_state()
                    state_lock.release()
                except Exception as e:
                    self._log(f"Paper trader save warning: {e}")

            if n_errors > 0:
                self._record_failure(
                    stage="daily_orchestrator",
                    error=f"{n_errors} ticker errors: " +
                          ", ".join(e.get("ticker", "?") for e in summary.get("errors", [])[:5])
                )

            self._log(f"Daily complete: {n_decisions} decisions, {n_errors} errors")
            return {"decisions": n_decisions, "errors": n_errors}
        except Exception as e:
            self._log(f"Daily orchestrator error: {e}")
            self._record_failure(stage="daily_orchestrator", error=str(e))
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
        """Auto-install cross-platform scheduler if not already present."""
        result = {"status": "skipped"}
        try:
            from schedule_pipeline import check_status, install_task

            ret = check_status()
            if ret == 0:
                self._log("Scheduler already installed")
                result["status"] = "already_installed"
                return result

            # Not installed - install it
            import platform
            scheduler_name = "Windows Task Scheduler" if platform.system() == "Windows" else "crontab"
            self._log(f"Installing {scheduler_name} (daily 4 PM)...")
            ret = install_task("16:00")
            result["status"] = "installed" if ret == 0 else "failed"
            result["returncode"] = ret
        except Exception as e:
            self._log(f"Scheduler auto-install failed: {e}")
            result["status"] = "error"
            result["error"] = str(e)
        return result

    def install_scheduler(self) -> dict:
        """Install cross-platform scheduler for daily 4 PM run."""
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

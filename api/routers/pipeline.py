"""Pipeline management endpoint."""

import logging
import threading
from fastapi import APIRouter
from src.models.model import models_exist
from src.data.data_fetcher import NSE_STOCKS

logger = logging.getLogger(__name__)

router = APIRouter()

_training_status = {}
_training_lock = threading.Lock()

_run_status = {"status": "idle", "result": None}
_run_lock = threading.Lock()


@router.get("/status")
def pipeline_status():
    try:
        trained = [t for t in NSE_STOCKS if models_exist(t)]
        missing = [t for t in NSE_STOCKS if not models_exist(t)]
        return {
            "total": len(NSE_STOCKS),
            "trained": len(trained),
            "missing": len(missing),
            "trained_tickers": trained,
            "missing_tickers": missing,
        }
    except Exception as e:
        return {"error": str(e)}


def _train_worker(ticker: str, force: bool):
    try:
        with _training_lock:
            _training_status[ticker] = {"status": "training", "progress": "Starting..."}
        from src.models.trainer import train_for_ticker
        result = train_for_ticker(ticker, force_retrain=force)
        with _training_lock:
            _training_status[ticker] = {"status": "done", "result": result}
    except Exception as e:
        with _training_lock:
            _training_status[ticker] = {"status": "error", "error": str(e)}


@router.post("/train/{ticker}")
def train_model(ticker: str, force: bool = False):
    try:
        with _training_lock:
            if ticker in _training_status and _training_status[ticker].get("status") == "training":
                return {"status": "already_training", "ticker": ticker}

        thread = threading.Thread(target=_train_worker, args=(ticker, force), daemon=True)
        thread.start()
        return {"status": "started", "ticker": ticker}
    except Exception as e:
        return {"error": str(e)}


@router.get("/train/{ticker}/status")
def train_status(ticker: str):
    with _training_lock:
        return _training_status.get(ticker, {"status": "idle"})


def _run_worker(tickers: list[str]):
    global _run_status
    try:
        with _run_lock:
            _run_status = {"status": "running", "result": None}

        from src.trading.ledger import Ledger
        from src.signals.orchestrator import DailyOrchestrator
        from src.trading.paper_trader import PaperTrader

        ledger = Ledger()
        trader = PaperTrader(initial_capital=200000)
        # Load existing state
        import os
        state_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_state.json")
        if os.path.exists(state_path):
            try:
                trader.load_state(state_path)
            except Exception:
                pass

        orch = DailyOrchestrator(tickers=tickers, ledger=ledger, paper_trader=trader)
        summary = orch.run()

        # Save paper trading state
        trader.save_state(state_path)
        ledger.close()

        with _run_lock:
            _run_status = {"status": "done", "result": summary}
        logger.info(f"Orchestrator run complete: {len(summary.get('decisions', []))} decisions, "
                     f"{len(summary.get('trades', []))} trades, {len(summary.get('errors', []))} errors")
    except Exception as e:
        logger.error(f"Orchestrator run failed: {e}")
        with _run_lock:
            _run_status = {"status": "error", "error": str(e)}


@router.post("/run")
def run_orchestrator(tickers: list[str] = None):
    try:
        with _run_lock:
            if _run_status.get("status") == "running":
                return {"status": "already_running"}

        if tickers is None:
            tickers = [t for t in NSE_STOCKS if models_exist(t)]
            if not tickers:
                return {"status": "no_trained_models", "message": "Train at least one model first"}
        else:
            # Validate user-supplied ticker list
            import re
            ticker_re = re.compile(r"^[A-Z0-9]{1,20}\.NS$")
            invalid = [t for t in tickers if not ticker_re.match(t)]
            if invalid:
                return {"error": f"Invalid ticker format: {invalid[:5]}"}
            if len(tickers) > 50:
                return {"error": "Too many tickers (max 50)"}

        thread = threading.Thread(target=_run_worker, args=(tickers,), daemon=True)
        thread.start()
        return {"status": "started", "tickers": tickers}
    except Exception as e:
        return {"error": str(e)}


@router.get("/run/status")
def run_status():
    with _run_lock:
        return _run_status

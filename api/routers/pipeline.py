"""Pipeline management endpoint."""

import threading
from fastapi import APIRouter
from src.model import models_exist
from src.data_fetcher import NSE_STOCKS

router = APIRouter()

_training_status = {}
_training_lock = threading.Lock()


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
        from src.trainer import train_for_ticker
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

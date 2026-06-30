"""Ledger / trading journal endpoint."""

import os
from fastapi import APIRouter, Query
from src.ledger import Ledger
from src.constants import LEDGER_DB

router = APIRouter()


def get_ledger():
    return Ledger(db_path=LEDGER_DB)


@router.get("/decisions")
def get_decisions(
    ticker: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
):
    try:
        ledger = get_ledger()
        decisions = ledger.get_decisions(ticker=ticker, start_date=start_date, end_date=end_date)
        ledger.close()
        return {"decisions": decisions}
    except Exception as e:
        return {"error": str(e)}


@router.get("/trades")
def get_trades(ticker: str = Query(None)):
    try:
        ledger = get_ledger()
        trades = ledger.get_trades(ticker=ticker)
        ledger.close()
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}


@router.get("/performance")
def get_performance():
    try:
        ledger = get_ledger()
        perf = ledger.get_performance()
        signal_acc = ledger.get_signal_accuracy()
        daily_pnl = ledger.get_daily_pnl()
        snapshots = ledger.get_snapshots()
        ledger.close()
        return {
            "performance": perf,
            "signal_accuracy": signal_acc,
            "daily_pnl": daily_pnl,
            "snapshots": snapshots,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/summary")
def ledger_summary():
    try:
        ledger = get_ledger()
        decisions = ledger.get_decisions()
        trades = ledger.get_trades()
        snapshots = ledger.get_snapshots()
        perf = ledger.get_performance()
        signal_acc = ledger.get_signal_accuracy()
        ledger.close()

        total_decisions = len(decisions)
        correct = sum(1 for d in decisions if d.get("correct"))
        accuracy = correct / total_decisions if total_decisions > 0 else 0

        return {
            "total_decisions": total_decisions,
            "total_trades": len(trades),
            "accuracy": round(accuracy, 4),
            "signal_accuracy": signal_acc,
            "snapshots": snapshots[-30:] if snapshots else [],
        }
    except Exception as e:
        return {"error": str(e)}

"""Ledger / trading journal endpoint."""

from fastapi import APIRouter, Query
from src.trading.ledger import Ledger
from src.core.constants import LEDGER_DB

router = APIRouter()


def get_ledger():
    return Ledger(db_path=LEDGER_DB)


@router.get("/decisions")
def get_decisions(
    ticker: str = Query(None),
    start_date: str = Query(None),
    end_date: str = Query(None),
    agreement: str = Query(None, description="'all' filters to decisions where the 3 main signals agreed"),
):
    ledger = get_ledger()
    try:
        decisions = ledger.get_decisions(
            ticker=ticker, start_date=start_date, end_date=end_date,
            main_signals_agree=(agreement == "all"),
        )
        return {"decisions": decisions}
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/agreement-stats")
def agreement_stats():
    """P4.2: is the system better when all main signals agree?"""
    ledger = get_ledger()
    try:
        return ledger.get_agreement_stats()
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/trades")
def get_trades(ticker: str = Query(None)):
    ledger = get_ledger()
    try:
        trades = ledger.get_trades(ticker=ticker)
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/performance")
def get_performance():
    ledger = get_ledger()
    try:
        perf = ledger.get_performance()
        signal_acc = ledger.get_signal_accuracy()
        daily_pnl = ledger.get_daily_pnl()
        snapshots = ledger.get_snapshots()
        return {
            "performance": perf,
            "signal_accuracy": signal_acc,
            "daily_pnl": daily_pnl,
            "snapshots": snapshots,
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/benchmark")
def get_benchmark():
    """Paper trading vs Nifty 50 buy-and-hold comparison (P3.2).

    The single most honest signal of whether the system adds value.
    """
    ledger = get_ledger()
    try:
        from src.signals.benchmarks import paper_vs_nifty
        return paper_vs_nifty(ledger)
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/calibration")
def get_calibration(source: str = Query("live")):
    """Confidence calibration report from resolved decisions (P4.4)."""
    ledger = get_ledger()
    try:
        from src.models.calibration import reliability_report
        decisions = ledger.get_decisions(source=source)
        return reliability_report(decisions)
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()


@router.get("/summary")
def ledger_summary():
    ledger = get_ledger()
    try:
        decisions = ledger.get_decisions()
        trades = ledger.get_trades()
        snapshots = ledger.get_snapshots()
        perf = ledger.get_performance()
        signal_acc = ledger.get_signal_accuracy()

        total_decisions = len(decisions)
        correct = sum(1 for d in decisions if d.get("correct"))
        accuracy = correct / total_decisions if total_decisions > 0 else 0

        return {
            "total_decisions": total_decisions,
            "total_trades": len(trades),
            "accuracy": round(accuracy, 4),
            "trade_accuracy": perf.get("trade_accuracy"),
            "hold_ratio": perf.get("hold_ratio"),
            "live_decisions": perf.get("live_decisions", 0),
            "backfill_decisions": perf.get("backfill_decisions", 0),
            "signal_accuracy": signal_acc,
            "snapshots": snapshots[-30:] if snapshots else [],
        }
    except Exception as e:
        return {"error": str(e)}
    finally:
        ledger.close()

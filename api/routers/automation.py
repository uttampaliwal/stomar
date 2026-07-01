from fastapi import APIRouter

from src.automation.daily_runner import DailyAutomationRunner
from src.trading.ledger import Ledger

router = APIRouter()


@router.post("/run")
def run_daily_cycle():
    try:
        runner = DailyAutomationRunner()
        summary = runner.run(dry_run=False)
        return {"ok": True, "summary": summary}
    except Exception as exc:  # pragma: no cover - defensive
        return {"ok": False, "error": str(exc)}


@router.get("/decisions")
def recent_decisions():
    try:
        ledger = Ledger()
        decisions = ledger.get_decisions(limit=20) if hasattr(ledger, "get_decisions") else []
        ledger.close()
        return {"decisions": decisions}
    except Exception as exc:  # pragma: no cover - defensive
        return {"error": str(exc)}

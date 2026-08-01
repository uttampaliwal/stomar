"""Hardware circuit breakers & risk control endpoints."""

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from services.risk_guard import risk_guard

router = APIRouter()


class ResumeRequest(BaseModel):
    override_token: str = Field(..., description="Manual override token to lift a drawdown freeze")


class EquityUpdate(BaseModel):
    equity: float = Field(..., gt=0, description="Current total portfolio equity")


@router.get("/status")
def guard_status():
    """Current breaker state: halt, drawdown, daily loss, allocation caps."""
    return risk_guard.get_status()


@router.post("/resume")
def resume_trading(req: ResumeRequest):
    """Manually lift a drawdown freeze with the override token."""
    result = risk_guard.resume_trading(req.override_token)
    if not result["resumed"]:
        raise HTTPException(status_code=403, detail=result["reason"])
    return result


@router.post("/equity")
def update_equity(req: EquityUpdate):
    """Mark-to-market update: re-evaluates daily loss and drawdown breakers."""
    risk_guard.update_equity(req.equity)
    return risk_guard.get_status()


@router.post("/pnl")
def record_pnl(pnl: float = Query(..., description="Realized P&L to add to the daily tracker")):
    """Record realized P&L for the daily loss limit."""
    risk_guard.record_realized_pnl(pnl)
    return risk_guard.get_status()


@router.get("/scalar")
def volatility_scalar(
    symbol: str | None = Query(None, description="Ticker for ATR expansion leg, e.g. RELIANCE.NS"),
    vix: float | None = Query(None, description="Override VIX level (default: live India VIX)"),
):
    """Volatility position-size scalar (VIX > 22 and/or ATR expansion > 2x)."""
    return risk_guard.get_volatility_scalar(symbol=symbol, vix=vix)

"""Portfolio analytics endpoint."""

from fastapi import APIRouter
from src.portfolio import Portfolio

router = APIRouter()

_portfolio = Portfolio(initial_capital=100000)


@router.get("/stats")
def portfolio_stats():
    try:
        stats = _portfolio.get_stats()
        return {"stats": stats}
    except Exception as e:
        return {"error": str(e)}


@router.get("/equity-curve")
def equity_curve():
    try:
        trades = _portfolio.trades if hasattr(_portfolio, "trades") else []
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}

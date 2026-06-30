"""Portfolio analytics endpoint."""

import os
from fastapi import APIRouter
from src.paper_trader import PaperTrader

router = APIRouter()

_trader = None


def _get_trader():
    global _trader
    if _trader is None:
        state_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_state.json")
        _trader = PaperTrader(initial_capital=200000)
        if os.path.exists(state_path):
            try:
                _trader.load_state(state_path)
            except Exception:
                pass
    return _trader


@router.get("/stats")
def portfolio_stats():
    try:
        trader = _get_trader()
        positions = []
        for ticker, p in trader.positions.items():
            if p.quantity != 0:
                positions.append({
                    "ticker": ticker,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 2),
                    "current_price": round(p.current_price, 2),
                    "pnl": round(p.pnl, 2),
                })

        return {
            "equity": round(trader.portfolio.equity, 2) if hasattr(trader.portfolio, "equity") else 0,
            "cash": round(trader.portfolio.cash, 2),
            "positions": positions,
            "realized_pnl": round(trader.realized_pnl, 2) if hasattr(trader, "realized_pnl") else 0,
            "unrealized_pnl": round(sum(p.pnl for p in trader.positions.values()), 2),
            "total_trades": len(trader.trade_log) if hasattr(trader, "trade_log") else 0,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/trades")
def trade_log():
    try:
        trader = _get_trader()
        trades = []
        for t in trader.trade_log[-50:]:
            trades.append({
                "ticker": t.ticker,
                "side": t.side.name if hasattr(t.side, "name") else str(t.side),
                "quantity": t.quantity,
                "price": round(t.price, 2),
                "date": str(t.date),
            })
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}

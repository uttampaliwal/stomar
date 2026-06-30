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
        equity = trader.get_equity()
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

        realized = sum(cp["pnl"] for cp in trader.closed_positions) if trader.closed_positions else 0

        return {
            "equity": round(equity, 2),
            "cash": round(trader.cash, 2),
            "positions": positions,
            "realized_pnl": round(realized, 2),
            "unrealized_pnl": round(sum(p.pnl for p in trader.positions.values()), 2),
            "total_trades": len(trader.trade_log),
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
                "price": round(t.fill_price, 2),
                "date": str(t.timestamp),
            })
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}

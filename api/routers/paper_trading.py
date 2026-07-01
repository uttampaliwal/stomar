"""Paper trading endpoint."""

import os
import json
from fastapi import APIRouter, Body
from src.trading.paper_trader import PaperTrader
from src.trading.engine import OrderSide, OrderType
from src.trading.risk_controls import RiskLimits
from src.data.data_fetcher import NSE_STOCKS

router = APIRouter()

_trader = None


def get_trader():
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


@router.get("/state")
def paper_state():
    try:
        trader = get_trader()
        state = trader.get_summary()
        return state
    except Exception as e:
        return {"error": str(e)}


@router.post("/order")
def place_order(data: dict = Body(...)):
    try:
        trader = get_trader()
        ticker = data.get("ticker", "")
        side_str = data.get("side", "BUY").upper()
        quantity = int(data.get("quantity", 0))

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

        from src.data.data_fetcher import get_live_price
        price = get_live_price(ticker)
        if price is None or price <= 0:
            return {"error": f"Cannot get price for {ticker}"}

        order = trader.place_order(ticker, side, OrderType.MARKET, quantity)
        return {
            "order_id": order.order_id if order else None,
            "status": "filled" if order else "rejected",
            "price": price,
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/positions")
def positions():
    try:
        trader = get_trader()
        pos = []
        for ticker, p in trader.positions.items():
            if p.quantity != 0:
                pos.append({
                    "ticker": ticker,
                    "quantity": p.quantity,
                    "avg_cost": round(p.avg_cost, 2),
                    "current_price": round(p.current_price, 2),
                    "market_value": round(p.market_value, 2),
                    "pnl": round(p.pnl, 2),
                    "pnl_pct": round(p.pnl_pct * 100, 2),
                })
        return {"positions": pos}
    except Exception as e:
        return {"error": str(e)}


@router.get("/trades")
def trade_log():
    try:
        trader = get_trader()
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


@router.get("/stocks")
def available_stocks():
    return {"stocks": NSE_STOCKS}

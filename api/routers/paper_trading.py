"""Paper trading endpoint."""

import os
import threading
from fastapi import APIRouter, Body
from src.trading.paper_trader import PaperTrader
from src.trading.engine import OrderSide, OrderType
from src.data.data_fetcher import NSE_STOCKS

router = APIRouter()

_trader = None
_trader_lock = threading.Lock()
_operation_lock = threading.Lock()  # Serializes order placement and reset


def get_trader():
    global _trader
    with _trader_lock:
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

        if quantity <= 0:
            return {"error": "Quantity must be greater than 0"}

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL

        from src.data.data_fetcher import get_live_price
        price = data.get("price")
        if not price or float(price) <= 0:
            price = get_live_price(ticker)
        else:
            price = float(price)

        if price is None or price <= 0:
            return {"error": f"Cannot get price for {ticker}"}

        with _operation_lock:
            order = trader.execute_market_trade(ticker, side, quantity, price=price)
            if order is None or order.status.name == "REJECTED":
                return {"order_id": order.order_id if order else None, "status": "rejected", "price": price}

            return {
                "order_id": order.order_id,
                "status": order.status.value.lower() if hasattr(order.status, 'value') else str(order.status).lower(),
                "price": price,
                "filled_quantity": order.filled_quantity,
                "filled_price": order.filled_price,
                "summary": trader.get_summary(),
            }
    except Exception as e:
        return {"error": str(e)}


@router.post("/reset")
def reset_paper_trading(data: dict = Body(default={})):
    try:
        trader = get_trader()
        capital = float(data.get("capital", 200000))
        with _operation_lock:
            trader.initial_capital = capital
            trader.reset()
            state_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "paper_state.json")
            trader.save_state(state_path)
        return {
            "status": "success",
            "message": f"Paper trader reset with capital {capital}",
            "state": trader.get_summary()
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
                "side": t.side,
                "quantity": t.quantity,
                "price": round(t.fill_price, 2),
                "fill_cost": round(t.fill_cost, 2),
                "pnl": round(t.pnl, 2),
                "date": str(t.timestamp),
            })
        return {"trades": trades}
    except Exception as e:
        return {"error": str(e)}


@router.get("/stocks")
def available_stocks():
    return {"stocks": NSE_STOCKS}


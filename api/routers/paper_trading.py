"""Paper trading endpoint.

Paper-only by design: this router can NEVER submit live orders. When the
system is in live mode, order placement here is refused and live orders
must flow through the broker-backed execution path instead.
"""

import re
import math
import threading
from contextlib import contextmanager
from fastapi import APIRouter, Body
from filelock import FileLock, Timeout
from src.trading.paper_trader import PaperTrader
from src.trading.engine import OrderSide, OrderType
from src.data.data_fetcher import NSE_STOCKS
from src.core.constants import PAPER_STATE_PATH
from src.core.trading_mode import get_trading_mode, mode_banner
from src.trading.execution_manager import ExecutionManager
from src.trading.risk_controls import RiskController, RiskLimits

router = APIRouter()

# Ticker must match NSE format: uppercase letters/digits, ending with .NS
_TICKER_RE = re.compile(r"^[A-Z0-9]{1,20}\.NS$")

_trader = None
_trader_lock = threading.Lock()
_operation_lock = threading.Lock()  # Serializes order placement and reset in-process

# Cross-process lock: gunicorn runs one module copy per worker, so in-memory
# singletons diverge. Every read-modify-write of paper_state.json must hold
# this lock across the full load → mutate → save span so workers stay coherent.
# The timeout keeps a long pipeline run from hanging UI requests.
_state_lock = FileLock(PAPER_STATE_PATH + ".lock", timeout=10)
_BUSY_MSG = "Paper trading state is busy (pipeline run or another operation in progress); retry shortly"
_manager: ExecutionManager | None = None


def get_manager() -> ExecutionManager:
    global _manager
    with _trader_lock:
        if _manager is None:
            _manager = ExecutionManager(
                broker=None,  # paper path uses PaperTrader directly; gate-only
                risk=RiskController(RiskLimits()),
                max_stale_quote_seconds=15.0,
                max_daily_orders=10,
            )
        return _manager


def get_trader():
    global _trader
    with _trader_lock:
        if _trader is None:
            _trader = PaperTrader(initial_capital=200000)
            _trader.load_state(PAPER_STATE_PATH)
        return _trader


def _refresh_trader(trader) -> None:
    """Reload the singleton from disk (cheap small JSON read).

    Without this, a worker would serve/mutate its own stale snapshot and
    clobber changes made by other workers.
    """
    try:
        with _state_lock:
            trader.load_state(PAPER_STATE_PATH)
    except Timeout:
        raise RuntimeError(_BUSY_MSG) from None


@contextmanager
def _paper_state_session(trader):
    """Serialized read-modify-write on the shared paper state.

    Holds the cross-process lock across load → yield → save, so concurrent
    workers serialize on disk and no worker can overwrite another's changes.
    """
    try:
        with _state_lock:
            trader.load_state(PAPER_STATE_PATH)
            yield
            trader.save_state(PAPER_STATE_PATH)
    except Timeout:
        raise RuntimeError(_BUSY_MSG) from None


@router.get("/performance")
def paper_performance():
    """P3.4: rolling accuracy, per-module signal accuracy, drawdown stats."""
    try:
        from src.trading.ledger import Ledger
        ledger = Ledger()
        try:
            return {
                "rolling_accuracy_30d": ledger.get_rolling_accuracy(days=30, source="live"),
                "signal_accuracy": ledger.get_signal_accuracy(source="live"),
                "drawdown": ledger.get_drawdown_stats(),
            }
        finally:
            ledger.close()
    except Exception as e:
        return {"error": str(e)}


@router.get("/state")
def paper_state():
    try:
        trader = get_trader()
        _refresh_trader(trader)
        state = trader.get_summary()
        mode = get_trading_mode()
        state["mode"] = mode.value
        state["mode_banner"] = mode_banner(mode)
        return state
    except Exception as e:
        return {"error": str(e)}


@router.post("/order")
def place_order(data: dict = Body(...)):
    try:
        # Paper router can never execute live orders.
        mode = get_trading_mode()
        if mode.value == "live":
            return {"error": "live mode active: paper order router disabled; "
                             "orders must flow through the broker execution path"}

        trader = get_trader()
        ticker = data.get("ticker", "")
        side_str = data.get("side", "BUY").upper()
        try:
            quantity = int(data.get("quantity", 0))
        except (TypeError, ValueError):
            return {"error": f"Invalid quantity: {data.get('quantity')!r} (expected an integer)"}
        order_type_str = data.get("order_type", "MARKET").upper()

        if not _TICKER_RE.match(ticker):
            return {"error": f"Invalid ticker format: {ticker!r} (expected e.g. RELIANCE.NS)"}

        if quantity <= 0 or quantity > 1_000_000:
            return {"error": "Quantity must be between 1 and 1,000,000"}

        side = OrderSide.BUY if side_str == "BUY" else OrderSide.SELL
        order_type = OrderType.MARKET if order_type_str == "MARKET" else OrderType.LIMIT

        from src.data.data_fetcher import get_live_price
        price = data.get("price")
        if not price or float(price) <= 0:
            price = get_live_price(ticker)
        else:
            price = float(price)

        if price is None or price <= 0:
            return {"error": f"Cannot get price for {ticker}"}

        limit_price = data.get("limit_price")
        if order_type == OrderType.LIMIT:
            if not limit_price or float(limit_price) <= 0:
                return {"error": "Limit price is required for LIMIT orders"}
            limit_price = float(limit_price)

        # Full execution gate before any paper order is placed.
        gate = get_manager().gate_order(
            ticker, side_str, quantity, order_value=price * quantity,
        )
        if not gate["approved"]:
            return {"error": f"Risk gate blocked order: {gate['reason']}"}

        with _operation_lock:
            with _paper_state_session(trader):
                # Fill any pending stops on this ticker first — the fresh
                # price may have crossed a stop level since the last on_bar()
                # for this ticker (no background price feed checks stops).
                trader.check_stops(ticker, price)

                order = trader.place_order(ticker, side, order_type, quantity, price=price,
                                           stop_price=limit_price or 0)
                if order is None or order.status.name == "REJECTED":
                    return {"order_id": order.order_id if order else None, "status": "rejected", "price": price}

                # For MARKET orders, fill immediately
                if order_type == OrderType.MARKET:
                    trader.on_bar(ticker, o=price, h=price, low=price, c=price, volume=1_000_000)

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
        try:
            capital = float(data.get("capital", 200000))
        except (TypeError, ValueError):
            return {"error": f"Invalid capital: {data.get('capital')!r} (expected a number)"}
        if not math.isfinite(capital) or capital <= 0 or capital > 100_000_000:
            return {"error": "Capital out of allowed range (1 to 100,000,000)"}
        with _operation_lock:
            with _paper_state_session(trader):
                trader.initial_capital = capital
                trader.reset()
        return {
            "status": "success",
            "message": f"Paper trader reset with capital {capital}",
            "state": trader.get_summary()
        }
    except Exception as e:
        return {"error": str(e)}


@router.post("/close-position")
def close_position(data: dict = Body(...)):
    try:
        ticker = data.get("ticker", "")
        if not _TICKER_RE.match(ticker):
            return {"error": f"Invalid ticker format: {ticker!r}"}

        trader = get_trader()
        if ticker not in trader.positions or trader.positions[ticker].quantity == 0:
            return {"error": f"No open position for {ticker}"}

        from src.data.data_fetcher import get_live_price
        price = get_live_price(ticker)
        if price is None or price <= 0:
            return {"error": f"Cannot get price for {ticker}"}

        with _operation_lock:
            with _paper_state_session(trader):
                record = trader.close_position(ticker, price=price)
                if record is None:
                    return {"error": f"Failed to close position for {ticker}"}
                return {
                    "status": "closed",
                    "ticker": ticker,
                    "price": price,
                    "pnl": round(record.pnl, 2),
                    "summary": trader.get_summary(),
                }
    except Exception as e:
        return {"error": str(e)}


@router.get("/positions")
def positions():
    try:
        trader = get_trader()
        _refresh_trader(trader)
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
        _refresh_trader(trader)
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


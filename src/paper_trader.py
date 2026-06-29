"""Paper Trading Mode.

Simulates live trading using real delayed prices without actual execution.
Maintains a virtual portfolio, tracks P&L, and generates a trade log.

Usage:
    trader = PaperTrader(initial_capital=100000)
    trader.place_order("RELIANCE.NS", OrderSide.BUY, OrderType.MARKET, 10)
    fills = trader.on_bar("RELIANCE.NS", open=2500, high=2520, low=2480, close=2510)
    summary = trader.get_summary()
"""

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.engine import (
    ExecutionEngine, Order, OrderSide, OrderType, OrderStatus,
    VolumeSlippage,
)
from src.risk_controls import RiskController, RiskLimits
logger = logging.getLogger(__name__)


@dataclass
class Position:
    ticker: str
    quantity: int = 0
    avg_cost: float = 0.0
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        if self.quantity > 0:
            return (self.current_price - self.avg_cost) * self.quantity
        elif self.quantity < 0:
            return (self.avg_cost - self.current_price) * abs(self.quantity)
        return 0.0

    @property
    def pnl_pct(self) -> float:
        cost_basis = abs(self.avg_cost * self.quantity)
        return (self.pnl / cost_basis) if cost_basis > 0 else 0.0


@dataclass
class PaperTradeRecord:
    timestamp: str
    ticker: str
    side: str
    order_type: str
    quantity: int
    fill_price: float
    fill_cost: float
    slippage: float
    pnl: float = 0.0
    cumulative_pnl: float = 0.0
    cash_after: float = 0.0


class PaperTrader:
    """Simulates trading with real delayed prices."""

    def __init__(self, initial_capital: float = 100_000,
                 slippage_bps: int = 5,
                 risk_limits: Optional[RiskLimits] = None):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: dict[str, Position] = {}
        self.closed_positions: list[dict] = []

        self.slippage_model = VolumeSlippage(
            base_rate=slippage_bps / 10_000,
            avg_volume=1_000_000,
        )
        self.engine = ExecutionEngine(slippage_model=self.slippage_model)
        self.risk_controller = RiskController(
            limits=risk_limits or RiskLimits(), initial_capital=initial_capital
        )
        self.trade_log: list[PaperTradeRecord] = []
        self.cumulative_pnl = 0.0

    def place_order(self, ticker: str, side: OrderSide, order_type: OrderType,
                    quantity: int, price: float = 0, stop_price: float = 0) -> Order:
        """Submit a paper order after risk checks."""
        order = Order(
            order_id="", ticker=ticker, side=side, order_type=order_type,
            quantity=quantity, price=price, stop_price=stop_price,
        )

        estimated_value = (price * quantity if price > 0 else
                           self._estimate_price(ticker, side) * quantity)
        holdings_value = sum(p.market_value for p in self.positions.values())

        risk = self.risk_controller.check_order(
            order_value=estimated_value,
            current_holdings_value=holdings_value,
            ticker=ticker, holdings=self.positions,
        )

        if not risk["approved"]:
            logger.warning("Order rejected by risk: %s", risk.get("reason", "multiple checks"))
            order.status = OrderStatus.REJECTED
            return order

        return self.engine.submit_order(order)

    def on_bar(self, ticker: str, o: float, h: float, low: float, c: float,
               volume: int = 1_000_000) -> list[PaperTradeRecord]:
        """Process a bar through the engine and record fills."""
        from src.engine import Bar
        bar = Bar(ticker, datetime.now().isoformat(), o, h, low, c, volume)
        filled_orders = self.engine.on_bar(bar)

        records = []
        for order in filled_orders:
            record = self._record_fill(order)
            records.append(record)

            # Update position
            self._update_position(order)

            # Update risk equity
            equity = self.get_equity()
            self.risk_controller.update_equity(equity)

        return records

    def _record_fill(self, order: Order) -> PaperTradeRecord:
        """Create a trade record from a filled order."""
        pnl = 0.0
        if order.side == OrderSide.SELL and order.ticker in self.positions:
            pos = self.positions[order.ticker]
            if pos.quantity > 0:
                pnl = (order.filled_price - pos.avg_cost) * order.filled_quantity
            else:
                pnl = (pos.avg_cost - order.filled_price) * order.filled_quantity
        elif order.side == OrderSide.BUY and order.ticker in self.positions:
            pos = self.positions[order.ticker]
            if pos.quantity < 0:
                pnl = (pos.avg_cost - order.filled_price) * min(order.filled_quantity, abs(pos.quantity))

        self.cumulative_pnl += pnl

        record = PaperTradeRecord(
            timestamp=datetime.now().isoformat(),
            ticker=order.ticker,
            side=order.side.value,
            order_type=order.order_type.value,
            quantity=order.filled_quantity,
            fill_price=order.filled_price,
            fill_cost=order.fill_cost,
            slippage=getattr(order, "_slippage", 0.0),
            pnl=pnl,
            cumulative_pnl=self.cumulative_pnl,
            cash_after=self.cash,
        )
        self.trade_log.append(record)
        return record

    def _update_position(self, order: Order):
        """Update portfolio position after a fill."""
        ticker = order.ticker

        if order.side == OrderSide.BUY:
            if ticker in self.positions and self.positions[ticker].quantity < 0:
                # Covering a short position
                pos = self.positions[ticker]
                cover_qty = min(order.filled_quantity, abs(pos.quantity))
                realized = (pos.avg_cost - order.filled_price) * cover_qty
                self.closed_positions.append({
                    "ticker": ticker,
                    "avg_cost": pos.avg_cost,
                    "sell_price": order.filled_price,
                    "quantity": cover_qty,
                    "pnl": realized,
                })
                self.cash -= order.fill_cost
                remaining = order.filled_quantity - cover_qty
                if remaining > 0:
                    # Flip to long
                    pos.quantity = remaining
                    pos.avg_cost = order.filled_price
                elif abs(pos.quantity) == cover_qty:
                    del self.positions[ticker]
                else:
                    pos.quantity += cover_qty
            else:
                # Opening/adding to long position
                if ticker not in self.positions:
                    self.positions[ticker] = Position(ticker=ticker)
                pos = self.positions[ticker]
                total_cost = pos.avg_cost * pos.quantity + order.filled_price * order.filled_quantity
                pos.quantity += order.filled_quantity
                pos.avg_cost = total_cost / pos.quantity if pos.quantity > 0 else 0
                self.cash -= order.fill_cost

        elif order.side == OrderSide.SELL:
            if ticker in self.positions and self.positions[ticker].quantity < 0:
                # Adding to short position
                pos = self.positions[ticker]
                old_qty = abs(pos.quantity)
                total_cost = pos.avg_cost * old_qty + order.filled_price * order.filled_quantity
                pos.quantity -= order.filled_quantity
                pos.avg_cost = total_cost / abs(pos.quantity)
                self.cash += order.fill_cost
            elif ticker in self.positions and self.positions[ticker].quantity > 0:
                # Closing/reducing long position
                pos = self.positions[ticker]
                realized = (order.filled_price - pos.avg_cost) * min(order.filled_quantity, pos.quantity)
                self.cash += order.fill_cost
                self.closed_positions.append({
                    "ticker": ticker,
                    "avg_cost": pos.avg_cost,
                    "sell_price": order.filled_price,
                    "quantity": min(order.filled_quantity, pos.quantity),
                    "pnl": realized,
                })
                if order.filled_quantity >= pos.quantity:
                    remaining = order.filled_quantity - pos.quantity
                    if remaining > 0:
                        # Flip to short
                        pos.quantity = -remaining
                        pos.avg_cost = order.filled_price
                    else:
                        del self.positions[ticker]
                else:
                    pos.quantity -= order.filled_quantity
            else:
                # Open a short position
                self.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=-order.filled_quantity,
                    avg_cost=order.filled_price,
                    current_price=order.filled_price,
                )
                self.cash += order.fill_cost

    def update_prices(self, prices: dict[str, float]):
        """Update current prices for all positions."""
        for ticker, price in prices.items():
            if ticker in self.positions:
                self.positions[ticker].current_price = price

    def get_equity(self) -> float:
        """Total equity = cash + positions market value."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    def get_summary(self) -> dict:
        """Get portfolio summary."""
        equity = self.get_equity()
        total_return = (equity - self.initial_capital) / self.initial_capital

        open_positions = {}
        for ticker, pos in self.positions.items():
            open_positions[ticker] = {
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "current_price": pos.current_price,
                "market_value": pos.market_value,
                "unrealized_pnl": pos.pnl,
                "unrealized_pnl_pct": pos.pnl_pct,
            }

        win_count = sum(1 for p in self.closed_positions if p["pnl"] > 0)
        total_closed = len(self.closed_positions)

        return {
            "initial_capital": self.initial_capital,
            "current_equity": equity,
            "cash": self.cash,
            "total_return_pct": total_return,
            "total_trades": len(self.trade_log),
            "open_positions": open_positions,
            "closed_positions": len(self.closed_positions),
            "realized_pnl": sum(p["pnl"] for p in self.closed_positions),
            "unrealized_pnl": sum(p.pnl for p in self.positions.values()),
            "win_rate": (win_count / total_closed) if total_closed > 0 else 0.0,
            "risk_status": self.risk_controller.get_status(),
        }

    def get_trade_log(self) -> list[dict]:
        """Get trade log as list of dicts."""
        return [
            {
                "timestamp": r.timestamp,
                "ticker": r.ticker,
                "side": r.side,
                "order_type": r.order_type,
                "quantity": r.quantity,
                "fill_price": r.fill_price,
                "fill_cost": r.fill_cost,
                "slippage": r.slippage,
                "pnl": r.pnl,
                "cumulative_pnl": r.cumulative_pnl,
                "cash_after": r.cash_after,
            }
            for r in self.trade_log
        ]

    def export_session(self, path: str = None):
        """Export session data to JSON."""
        if path is None:
            from src.constants import DATA_DIR
            path = os.path.join(DATA_DIR, "paper_session.json")
        data = {
            "summary": self.get_summary(),
            "trade_log": self.get_trade_log(),
            "closed_positions": self.closed_positions,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Session exported to %s", path)

    def save_state(self, path: str = None):
        """Save paper trading state to disk for persistence across restarts."""
        if path is None:
            from src.constants import PAPER_STATE_PATH
            path = PAPER_STATE_PATH
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "cumulative_pnl": self.cumulative_pnl,
            "positions": {
                t: {"quantity": p.quantity, "avg_cost": p.avg_cost,
                    "current_price": p.current_price}
                for t, p in self.positions.items()
            },
            "closed_positions": self.closed_positions,
            "trade_log": self.get_trade_log(),
            "risk_daily_pnl": self.risk_controller.daily_pnl,
            "risk_weekly_pnl": self.risk_controller.weekly_pnl,
            "risk_peak_equity": self.risk_controller.peak_equity,
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("State saved to %s", path)

    def load_state(self, path: str = None) -> bool:
        """Load paper trading state from disk. Returns True if loaded."""
        if path is None:
            from src.constants import PAPER_STATE_PATH
            path = PAPER_STATE_PATH
        import os
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)
        self.cash = state.get("cash", self.initial_capital)
        self.cumulative_pnl = state.get("cumulative_pnl", 0.0)
        self.positions = {
            t: Position(ticker=t, quantity=v["quantity"], avg_cost=v["avg_cost"],
                        current_price=v["current_price"])
            for t, v in state.get("positions", {}).items()
        }
        self.closed_positions = state.get("closed_positions", [])
        self.risk_controller.daily_pnl = state.get("risk_daily_pnl", 0.0)
        self.risk_controller.weekly_pnl = state.get("risk_weekly_pnl", 0.0)
        self.risk_controller.peak_equity = state.get(
            "risk_peak_equity", self.initial_capital)
        self.risk_controller.current_equity = self.get_equity()
        logger.info("State loaded from %s", path)
        return True

    def _estimate_price(self, ticker: str, side: OrderSide) -> float:
        """Estimate price for risk check if not yet available."""
        if ticker in self.positions:
            return self.positions[ticker].current_price
        return 100.0  # Default fallback

    def reset(self):
        """Reset all state for a new session."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.closed_positions.clear()
        self.trade_log.clear()
        self.cumulative_pnl = 0.0
        self.engine = ExecutionEngine(slippage_model=self.slippage_model)
        self.risk_controller = RiskController(
            limits=self.risk_controller.limits, initial_capital=self.initial_capital
        )

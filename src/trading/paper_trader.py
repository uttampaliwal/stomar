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
import tempfile
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.trading.engine import (
    ExecutionEngine, Order, OrderSide, OrderType, OrderStatus,
    VolumeSlippage,
)
from src.trading.risk_controls import RiskController, RiskLimits
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

        is_closing = False
        if ticker in self.positions:
            pos = self.positions[ticker]
            if (side == OrderSide.SELL and pos.quantity > 0) or (side == OrderSide.BUY and pos.quantity < 0):
                is_closing = True

        risk = self.risk_controller.check_order(
            order_value=estimated_value,
            current_holdings_value=holdings_value,
            ticker=ticker, holdings=self.positions,
            is_closing=is_closing,
        )

        if not risk["approved"]:
            logger.warning("Order rejected by risk: %s", risk.get("reason", "multiple checks"))
            order.status = OrderStatus.REJECTED
            return order

        return self.engine.submit_order(order)

    def close_position(self, ticker: str, price: float = 0) -> Optional[PaperTradeRecord]:
        """Close an entire position at the given price.

        Returns the trade record, or None if no position exists.
        """
        if ticker not in self.positions or self.positions[ticker].quantity == 0:
            return None

        pos = self.positions[ticker]
        qty = abs(pos.quantity)
        side = OrderSide.SELL if pos.quantity > 0 else OrderSide.BUY

        if price <= 0:
            price = pos.current_price
        if price <= 0:
            logger.warning("Cannot close %s: no valid price", ticker)
            return None

        order = self.execute_market_trade(ticker, side, qty, price=price)
        if order.status == OrderStatus.REJECTED:
            logger.warning("Close order rejected for %s", ticker)
            return None

        return self.trade_log[-1] if self.trade_log else None

    def execute_market_trade(self, ticker: str, side: OrderSide, quantity: int, price: float) -> Order:
        """Submit and immediately fill a market order using specified live price."""
        order = self.place_order(ticker, side, OrderType.MARKET, quantity, price=price)
        if order.status == OrderStatus.REJECTED:
            return order

        # Immediate bar execution
        self.on_bar(ticker, o=price, h=price, low=price, c=price, volume=1_000_000)
        try:
            self.save_state()
        except Exception as e:
            logger.warning("Failed to auto-save paper trading state: %s", e)
        return order


    def on_bar(self, ticker: str, o: float, h: float, low: float, c: float,
               volume: int = 1_000_000) -> list[PaperTradeRecord]:
        """Process a bar through the engine and record fills."""
        from src.trading.engine import Bar
        bar = Bar(ticker, datetime.now().isoformat(), o, h, low, c, volume)
        if ticker in self.positions:
            self.positions[ticker].current_price = c

        filled_orders = self.engine.on_bar(bar)

        records = []
        for order in filled_orders:
            record = self._record_fill(order)
            records.append(record)

            # Update position
            self._update_position(order)
            if ticker in self.positions:
                self.positions[ticker].current_price = order.filled_price

            # Update risk equity
            equity = self.get_equity()
            self.risk_controller.update_equity(equity)

        return records

    def _record_fill(self, order: Order) -> PaperTradeRecord:
        """Create a trade record from a filled order."""
        pnl = 0.0
        if order.side == OrderSide.SELL and order.ticker in self.positions:
            pos = self.positions[order.ticker]
            sell_costs = order.fill_cost - order.filled_price * order.filled_quantity
            if pos.quantity > 0:
                pnl = (order.filled_price - pos.avg_cost) * order.filled_quantity - sell_costs
            else:
                pnl = (pos.avg_cost - order.filled_price) * order.filled_quantity - sell_costs
        elif order.side == OrderSide.BUY and order.ticker in self.positions:
            pos = self.positions[order.ticker]
            if pos.quantity < 0:
                buy_costs = order.fill_cost - order.filled_price * order.filled_quantity
                pnl = (pos.avg_cost - order.filled_price) * min(order.filled_quantity, abs(pos.quantity)) - buy_costs

        self.cumulative_pnl += pnl

        # Circuit breaker: only closing trades (pnl != 0) count as outcomes.
        # Entry fills (pnl == 0) must not reset the losing streak.
        if pnl != 0.0:
            self.risk_controller.update_consecutive_losses(pnl < 0)

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
                trade_costs = max(0.0, order.fill_cost - order.filled_price * order.filled_quantity)
                realized = (pos.avg_cost - order.filled_price) * cover_qty - trade_costs
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
                pos.avg_cost = total_cost / abs(pos.quantity) if pos.quantity != 0 else 0
                trade_costs = max(0.0, order.fill_cost - order.filled_price * order.filled_quantity)
                net_proceeds = order.filled_price * order.filled_quantity - trade_costs
                self.cash += net_proceeds
            elif ticker in self.positions and self.positions[ticker].quantity > 0:
                # Closing/reducing long position
                pos = self.positions[ticker]
                close_qty = min(order.filled_quantity, pos.quantity)
                trade_costs = max(0.0, order.fill_cost - order.filled_price * order.filled_quantity)
                realized = (order.filled_price - pos.avg_cost) * close_qty - trade_costs
                net_proceeds = order.filled_price * order.filled_quantity - trade_costs
                self.cash += net_proceeds
                self.closed_positions.append({
                    "ticker": ticker,
                    "avg_cost": pos.avg_cost,
                    "sell_price": order.filled_price,
                    "quantity": close_qty,
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
                trade_costs = max(0.0, order.fill_cost - order.filled_price * order.filled_quantity)
                net_proceeds = order.filled_price * order.filled_quantity - trade_costs
                self.cash += net_proceeds

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
        """Export session data to JSON atomically."""
        if path is None:
            from src.core.constants import DATA_DIR
            path = os.path.join(DATA_DIR, "paper_session.json")
        data = {
            "summary": self.get_summary(),
            "trade_log": self.get_trade_log(),
            "closed_positions": self.closed_positions,
        }
        dir_name = os.path.dirname(path) or "."
        fd, tmp_path = tempfile.mkstemp(dir=dir_name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, path)
            logger.info("Session exported to %s", path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def save_state(self, path: str = None):
        """Save paper trading state to disk for persistence across restarts."""
        if path is None:
            from src.core.constants import PAPER_STATE_PATH
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
            "engine": self.engine.get_state(),
        }
        import tempfile
        dir_name = os.path.dirname(path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(dir=dir_name or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(state, f, indent=2, default=str)
            os.replace(tmp_path, path)
            logger.info("State saved to %s", path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

    def load_state(self, path: str = None) -> bool:
        """Load paper trading state from disk. Returns True if loaded."""
        if path is None:
            from src.core.constants import PAPER_STATE_PATH
            path = PAPER_STATE_PATH
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupted paper state file %s: %s", path, e)
            return False

        if not isinstance(state, dict):
            logger.warning("Invalid paper state format in %s — expected dict", path)
            return False

        self.cash = state.get("cash", self.initial_capital)
        self.cumulative_pnl = state.get("cumulative_pnl", 0.0)
        self.positions = {
            t: Position(ticker=t, quantity=v["quantity"], avg_cost=v["avg_cost"],
                        current_price=v["current_price"])
            for t, v in state.get("positions", {}).items()
            if isinstance(v, dict) and "quantity" in v and "avg_cost" in v
        }
        self.closed_positions = state.get("closed_positions", [])
        raw_log = state.get("trade_log", [])
        self.trade_log = [
            PaperTradeRecord(
                timestamp=r.get("timestamp", ""),
                ticker=r.get("ticker", ""),
                side=r.get("side", ""),
                order_type=r.get("order_type", ""),
                quantity=r.get("quantity", 0),
                fill_price=r.get("fill_price", 0.0),
                fill_cost=r.get("fill_cost", 0.0),
                slippage=r.get("slippage", 0.0),
                pnl=r.get("pnl", 0.0),
                cumulative_pnl=r.get("cumulative_pnl", 0.0),
                cash_after=r.get("cash_after", 0.0),
            )
            for r in raw_log
        ]
        self.risk_controller.daily_pnl = state.get("risk_daily_pnl", 0.0)
        self.risk_controller.weekly_pnl = state.get("risk_weekly_pnl", 0.0)
        self.risk_controller.peak_equity = state.get(
            "risk_peak_equity", self.initial_capital)
        self.risk_controller.current_equity = self.get_equity()

        engine_state = state.get("engine")
        if engine_state:
            self.engine.restore_state(engine_state)

        logger.info("State loaded from %s", path)
        return True

    def _estimate_price(self, ticker: str, side: OrderSide) -> float:
        """Estimate price for risk check if not yet available."""
        if ticker in self.positions:
            return self.positions[ticker].current_price
        logger.debug("No price available for %s, using 0 for risk check", ticker)
        return 0.0

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

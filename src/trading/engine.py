"""Event-driven execution engine for backtest-live parity.

Processes orders bar-by-bar, just like live trading.
Same code path for backtest, paper, and live modes.

Usage:
    from src.trading.engine import ExecutionEngine, Order, OrderSide, OrderType, Bar

    engine = ExecutionEngine()
    order = Order(order_id="", ticker="RELIANCE.NS", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=10)
    engine.submit_order(order)
    filled = engine.on_bar(Bar("RELIANCE.NS", "2025-01-01", 2500, 2520, 2480, 2510, 1000000))
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"


class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_MARKET = "STOP_MARKET"


class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class Order:
    """Trading order."""
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0
    stop_price: float = 0.0
    status: OrderStatus = OrderStatus.PENDING
    filled_price: float = 0.0
    filled_quantity: int = 0
    fill_cost: float = 0.0
    timestamp: str = ""
    notes: str = ""


@dataclass
class Bar:
    """Single OHLCV bar."""
    ticker: str
    timestamp: str
    open: float
    high: float
    low: float
    close: float
    volume: float


class ExecutionEngine:
    """Event-driven execution engine.

    Processes one bar at a time, checks pending orders against new
    price data, fills orders based on order type and price action.
    """

    def __init__(self, slippage_model=None, fill_probability: float = 1.0,
                 rng: "np.random.Generator" = None):
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.rejected_orders: list[Order] = []
        self.order_counter = 0
        self.slippage_model = slippage_model or FixedSlippage(0.001)
        self.fill_probability = fill_probability
        # Per-instance seeded RNG so limit-order fills are reproducible across
        # runs (backtests, hyperparameter comparisons, regression debugging).
        # Callers may inject their own Generator to chain experiments.
        self.rng = rng if rng is not None else np.random.default_rng(42)

    def submit_order(self, order: Order) -> Order:
        """Submit an order to the engine."""
        self.order_counter += 1
        order.order_id = f"ORD-{self.order_counter:06d}"
        order.status = OrderStatus.PENDING
        self.pending_orders.append(order)
        logger.info(f"Order submitted: {order.order_id} {order.side.value} "
                    f"{order.quantity} {order.ticker} @ {order.order_type.value}")
        return order

    def on_bar(self, bar: Bar) -> list[Order]:
        """Process a new bar. Returns list of filled orders."""
        # Validate bar prices
        if bar.high < bar.low:
            logger.warning(f"Invalid bar for {bar.ticker}: high ({bar.high}) < low ({bar.low})")
            return []
        if bar.close <= 0 or bar.open <= 0:
            logger.warning(f"Invalid bar for {bar.ticker}: non-positive price")
            return []

        filled = []
        remaining = []

        for order in self.pending_orders:
            if order.ticker != bar.ticker:
                remaining.append(order)
                continue

            fill_price = self._check_fill(order, bar)

            if fill_price is not None:
                prob = self._estimate_fill_probability(order, bar)
                if self.rng.random() > prob:
                    remaining.append(order)
                    continue

                slippage = self.slippage_model.calculate(
                    order.side, fill_price, bar.volume
                )
                actual_fill = fill_price + slippage if order.side == OrderSide.BUY \
                    else fill_price - slippage

                from src.core.constants import calculate_nse_costs
                costs = calculate_nse_costs(
                    actual_fill, order.quantity,
                    "buy" if order.side == OrderSide.BUY else "sell",
                )

                order.filled_price = actual_fill
                order.filled_quantity = order.quantity
                order.fill_cost = actual_fill * order.quantity + costs["total"]
                order.status = OrderStatus.FILLED
                order.notes = (f"Fill: {actual_fill:.2f}, "
                               f"costs: {costs['total']:.2f}")

                self.filled_orders.append(order)
                filled.append(order)
                logger.info(f"Order filled: {order.order_id} @ {actual_fill:.2f}")
            else:
                remaining.append(order)

        self.pending_orders = remaining
        return filled

    def _check_fill(self, order: Order, bar: Bar) -> Optional[float]:
        """Check if an order would fill on this bar."""
        if order.order_type == OrderType.MARKET:
            return bar.open

        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and bar.low <= order.price:
                return min(order.price, bar.open)
            elif order.side == OrderSide.SELL and bar.high >= order.price:
                return max(order.price, bar.open)

        elif order.order_type == OrderType.STOP_LOSS:
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                return order.stop_price
            elif order.side == OrderSide.BUY and bar.high >= order.stop_price:
                return order.stop_price

        elif order.order_type == OrderType.STOP_MARKET:
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                return bar.open
            elif order.side == OrderSide.BUY and bar.high >= order.stop_price:
                return bar.open

        return None

    def _estimate_fill_probability(self, order: Order, bar: Bar) -> float:
        """Estimate fill probability. Market/Stop=100%, limit depends on distance."""
        if order.order_type == OrderType.MARKET:
            return 1.0
        if order.order_type in (OrderType.STOP_LOSS, OrderType.STOP_MARKET):
            return 1.0

        if order.order_type == OrderType.LIMIT and bar.close > 0:
            distance = abs(order.price - bar.close) / bar.close
            if distance < 0.005:
                prob = 0.9
            elif distance < 0.01:
                prob = 0.7
            elif distance < 0.02:
                prob = 0.4
            else:
                prob = 0.1
            return min(prob, self.fill_probability)

        return self.fill_probability

    def cancel_all(self, ticker: str = None) -> list[Order]:
        """Cancel all pending orders."""
        cancelled = []
        remaining = []
        for order in self.pending_orders:
            if ticker and order.ticker != ticker:
                remaining.append(order)
                continue
            order.status = OrderStatus.CANCELLED
            cancelled.append(order)
        self.pending_orders = remaining
        self.rejected_orders.extend(cancelled)
        return cancelled

    def get_pending(self, ticker: str = None) -> list:
        if ticker:
            return [o for o in self.pending_orders if o.ticker == ticker]
        return self.pending_orders

    def get_filled(self, ticker: str = None) -> list:
        if ticker:
            return [o for o in self.filled_orders if o.ticker == ticker]
        return self.filled_orders

    def get_state(self) -> dict:
        """Serialize engine state for persistence."""
        return {
            "order_counter": self.order_counter,
            "pending_orders": [self._order_to_dict(o) for o in self.pending_orders],
            "filled_orders": [self._order_to_dict(o) for o in self.filled_orders],
            "rejected_orders": [self._order_to_dict(o) for o in self.rejected_orders],
        }

    def restore_state(self, state: dict):
        """Restore engine state from serialized dict."""
        self.order_counter = state.get("order_counter", 0)
        self.pending_orders = [self._order_from_dict(d) for d in state.get("pending_orders", [])]
        self.filled_orders = [self._order_from_dict(d) for d in state.get("filled_orders", [])]
        self.rejected_orders = [self._order_from_dict(d) for d in state.get("rejected_orders", [])]

    @staticmethod
    def _order_to_dict(order: Order) -> dict:
        return {
            "order_id": order.order_id,
            "ticker": order.ticker,
            "side": order.side.value,
            "order_type": order.order_type.value,
            "quantity": order.quantity,
            "price": order.price,
            "stop_price": order.stop_price,
            "status": order.status.value,
            "filled_price": order.filled_price,
            "filled_quantity": order.filled_quantity,
            "fill_cost": order.fill_cost,
            "timestamp": order.timestamp,
            "notes": order.notes,
        }

    @staticmethod
    def _order_from_dict(d: dict) -> Order:
        return Order(
            order_id=d["order_id"],
            ticker=d["ticker"],
            side=OrderSide(d["side"]),
            order_type=OrderType(d["order_type"]),
            quantity=d["quantity"],
            price=d.get("price", 0.0),
            stop_price=d.get("stop_price", 0.0),
            status=OrderStatus(d.get("status", "PENDING")),
            filled_price=d.get("filled_price", 0.0),
            filled_quantity=d.get("filled_quantity", 0),
            fill_cost=d.get("fill_cost", 0.0),
            timestamp=d.get("timestamp", ""),
            notes=d.get("notes", ""),
        )


# ─── Slippage Models ───

class FixedSlippage:
    """Fixed percentage slippage."""
    def __init__(self, rate: float = 0.001):
        self.rate = rate

    def calculate(self, side: OrderSide, price: float, volume: float) -> float:
        return price * self.rate


class VolumeSlippage:
    """Volume-based: higher volume = lower slippage."""
    def __init__(self, base_rate: float = 0.002, avg_volume: float = 1_000_000):
        self.base_rate = base_rate
        self.avg_volume = avg_volume

    def calculate(self, side: OrderSide, price: float, volume: float) -> float:
        vol_ratio = self.avg_volume / max(volume, 1)
        rate = self.base_rate * min(vol_ratio, 3.0)
        return price * rate


class AdaptiveSlippage:
    """Slippage that increases with order size relative to volume."""
    def __init__(self, base_rate: float = 0.001, impact_factor: float = 0.1):
        self.base_rate = base_rate
        self.impact_factor = impact_factor

    def calculate(self, side: OrderSide, price: float, volume: float,
                  order_value: float = 0) -> float:
        base = price * self.base_rate
        impact = (order_value / max(volume * price, 1)) * self.impact_factor * price
        return base + impact


class BacktestExecutionSimulator:
    """Realistic execution simulation for backtesting.

    Models volume-based slippage, market impact, order type simulation,
    and fill probability for limit orders.
    """

    def __init__(self, slippage_model="volume", impact_factor: float = 0.1):
        self.slippage_model = slippage_model
        self.impact_factor = impact_factor

    def simulate_fill(self, side: OrderSide, price: float, volume: float,
                      order_value: float = 0) -> float:
        """Simulate realistic fill with slippage and impact."""
        if self.slippage_model == "volume":
            model = VolumeSlippage()
        elif self.slippage_model == "adaptive":
            model = AdaptiveSlippage(impact_factor=self.impact_factor)
        else:
            model = FixedSlippage()

        slippage = model.calculate(side, price, volume)
        return price + slippage if side == OrderSide.BUY else price - slippage

    def estimate_fill_probability(self, order_type: OrderType, price: float,
                                  bar_close: float) -> float:
        """Estimate probability that a limit order fills."""
        if order_type == OrderType.MARKET:
            return 1.0
        if order_type in (OrderType.STOP_LOSS, OrderType.STOP_MARKET):
            return 1.0
        if order_type == OrderType.LIMIT and bar_close > 0:
            distance = abs(price - bar_close) / bar_close
            if distance < 0.005:
                return 0.9
            elif distance < 0.01:
                return 0.7
            elif distance < 0.02:
                return 0.4
            return 0.1
        return 0.5

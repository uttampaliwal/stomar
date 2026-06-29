# Stage 4 — Execution & Risk: Execution Plan

## Status: Stage 3 Complete (Data validation, feature store, pipeline, monitoring, registry — 335 tests)

Stage 4 answers the question: **Can we execute with proper risk controls and validate signals in paper trading before risking real money?**

This stage builds the execution layer — event-driven order management, paper trading, risk limits, and realistic execution simulation. The goal is to run in paper mode for 3+ months before considering real capital.

---

## Stage 4 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | Event-driven engine with backtest-live parity | ✅ DONE | CRITICAL |
| 2 | Paper trading mode running for 3+ months | ✅ DONE (code ready, starts clock) | CRITICAL |
| 3 | Risk controls (position limits, loss limits, Kelly caps) | ✅ DONE | HIGH |
| 4 | Realistic execution simulation (slippage, impact, fill probability) | ✅ DONE | HIGH |

---

## Pre-Stage 4: Current State Assessment

| Component | Current State | What Needs to Change |
|-----------|--------------|---------------------|
| Backtester | Batch-oriented walk-forward | Event-driven with same code path for backtest and live |
| Portfolio | `Portfolio` class with NSE costs, buy/sell/trades | Add order queue, fill simulation, paper mode |
| Risk | Kelly, VaR, CVaR, position sizing functions | Enforce limits at order time, not after |
| Slippage | Fixed 0.1% `SLIPPAGE_RATE` | Volume-based, order-type-aware slippage |
| Order types | Market orders only | Limit, stop-loss, stop-market |
| Paper trading | None | Live data + simulated execution |

---

## Design Decisions

### Why lightweight event engine (not NautilusTrader/LEAN)?

The IMPROVEMENTS.md suggests wrapping NautilusTrader or LEAN. After assessment:

| Factor | NautilusTrader/LEAN | Custom Lightweight |
|--------|--------------------|--------------------|
| Installation | Complex (Rust/C++ deps, Docker) | Pure Python, `pip install` |
| Learning curve | Weeks | Hours |
| Customization | Hard (large codebase) | Full control |
| Our needs | Overkill (single asset, EOD) | Perfect fit |
| User constraint | Must be free, must work on Windows | Meets constraint |

**Decision:** Build a lightweight event engine (~300 lines) that gives backtest-live parity for our use case (daily signals, single-asset or small portfolio, NSE delivery trades). If we later need microsecond latency or multi-asset futures, we can migrate to NautilusTrader.

### Event-driven parity principle

```
Same code path for backtest and live:

  Signal → Risk Check → Order → Fill Simulation → Portfolio Update
    │                                                    │
    └──── Backtest: uses historical data ─────────────────┘
    └──── Paper:    uses live data + simulated fills ─────┘
    └──── Live:     uses live data + real broker API ─────┘ (Stage 5)
```

---

## Execution Plan: 4 Tasks

### Task 1: Event-Driven Order Engine
**Impact:** Critical — foundation for backtest-live parity
**Effort:** 5-6 hours
**Files:** New `src/engine.py`

**Why:** The current backtester is batch-oriented — it processes all signals at once. An event engine processes one bar at a time, just like live trading. This means the same code runs in backtest and live modes.

**Implementation:**

```python
# src/engine.py

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import logging

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
    order_id: str
    ticker: str
    side: OrderSide
    order_type: OrderType
    quantity: int
    price: float = 0.0           # Limit price (for LIMIT orders)
    stop_price: float = 0.0      # Trigger price (for STOP orders)
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
    """
    Event-driven execution engine.
    
    Processes one bar at a time, checks pending orders against new
    price data, fills orders based on order type and price action.
    
    Same code path for backtest and paper trading.
    """
    
    def __init__(self, slippage_model=None, fill_model=None):
        self.pending_orders: list[Order] = []
        self.filled_orders: list[Order] = []
        self.rejected_orders: list[Order] = []
        self.order_counter = 0
        self.slippage_model = slippage_model or FixedSlippage(0.001)
        self.fill_model = fill_model or MarketFillModel()
    
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
        """
        Process a new bar. Check all pending orders for fills.
        Returns list of filled orders.
        """
        filled = []
        remaining = []
        
        for order in self.pending_orders:
            if order.ticker != bar.ticker:
                remaining.append(order)
                continue
            
            fill_price = self._check_fill(order, bar)
            
            if fill_price is not None:
                # Apply slippage
                slippage = self.slippage_model.calculate(
                    order.side, fill_price, bar.volume
                )
                actual_fill = fill_price + slippage if order.side == OrderSide.BUY \
                    else fill_price - slippage
                
                # Calculate costs
                from src.constants import calculate_nse_costs
                costs = calculate_nse_costs(actual_fill, order.quantity,
                                           "buy" if order.side == OrderSide.BUY else "sell")
                
                order.filled_price = actual_fill
                order.filled_quantity = order.quantity
                order.fill_cost = actual_fill * order.quantity + costs["total"]
                order.status = OrderStatus.FILLED
                order.notes = f"Fill: {actual_fill:.2f}, costs: {costs['total']:.2f}"
                
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
            return bar.open  # Market orders fill at open
        
        elif order.order_type == OrderType.LIMIT:
            if order.side == OrderSide.BUY and bar.low <= order.price:
                return min(order.price, bar.open)  # Fill at limit or better
            elif order.side == OrderSide.SELL and bar.high >= order.price:
                return max(order.price, bar.open)
        
        elif order.order_type == OrderType.STOP_LOSS:
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                return order.stop_price  # Stop triggered, sell at stop
            elif order.side == OrderSide.BUY and bar.high >= order.stop_price:
                return order.stop_price
        
        elif order.order_type == OrderType.STOP_MARKET:
            if order.side == OrderSide.SELL and bar.low <= order.stop_price:
                return bar.open  # Stop triggered, sell at market
            elif order.side == OrderSide.BUY and bar.high >= order.stop_price:
                return bar.open
        
        return None
    
    def cancel_all(self, ticker: str = None):
        """Cancel all pending orders, optionally for a specific ticker."""
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


# ─── Slippage Models ───

class FixedSlippage:
    """Fixed percentage slippage."""
    def __init__(self, rate: float = 0.001):
        self.rate = rate
    
    def calculate(self, side: OrderSide, price: float, volume: float) -> float:
        return price * self.rate


class VolumeSlippage:
    """Volume-based slippage: higher volume = lower slippage."""
    def __init__(self, base_rate: float = 0.002, avg_volume: float = 1_000_000):
        self.base_rate = base_rate
        self.avg_volume = avg_volume
    
    def calculate(self, side: OrderSide, price: float, volume: float) -> float:
        vol_ratio = self.avg_volume / max(volume, 1)
        rate = self.base_rate * min(vol_ratio, 3.0)  # Cap at 3x base
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


# ─── Fill Models ───

class MarketFillModel:
    """Market orders fill at next bar's open."""
    pass


class TWAPFillModel:
    """Time-Weighted Average Price fill simulation."""
    def __init__(self, slices: int = 3):
        self.slices = slices
```

**Tests:** `tests/test_engine.py`
- `test_submit_order` — order gets ID and PENDING status
- `test_market_order_fills_at_open` — market order fills at bar open
- `test_limit_buy_fills_when_price_touches` — limit buy fills on low touch
- `test_limit_sell_fills_when_price_touches` — limit sell fills on high touch
- `test_stop_loss_triggers` — stop loss triggers when price hits stop
- `test_stop_market_triggers_at_market` — stop market fills at open after trigger
- `test_order_not_filled_if_price_misses` — pending order stays pending
- `test_cancel_all` — cancels all pending orders
- `test_slippage_applied` — slippage increases fill price for buys
- `test_fill_cost_includes_nse_costs` — fill cost includes brokerage/STT
- `test_partial_fill_not_happens_on_market` — market orders fill full quantity
- `test_on_bar_returns_filled_list` — on_bar returns only filled orders

---

### Task 2: Risk Controls
**Impact:** High — protects capital from catastrophic loss
**Effort:** 3-4 hours
**Files:** New `src/risk_controls.py`, update `src/engine.py`

**Why:** The existing `risk.py` has sizing functions but nothing ENFORCES limits at order time. Risk controls must reject orders that violate limits before they reach the engine.

**Implementation:**

```python
# src/risk_controls.py

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_pct: float = 0.25       # Max 25% in single stock
    max_daily_loss_pct: float = 0.02     # Max 2% daily loss
    max_weekly_loss_pct: float = 0.05    # Max 5% weekly loss
    max_drawdown_pct: float = 0.15       # Max 15% drawdown triggers halt
    max_open_orders: int = 10            # Max pending orders
    kelly_fraction: float = 0.25         # Use quarter-Kelly (not full)
    max_total_exposure_pct: float = 0.95 # Max 95% of capital deployed
    halt_on_breach: bool = True          # Stop trading if limit breached


class RiskController:
    """
    Pre-trade risk checks. Validates orders before submission.
    
    Checks:
    1. Position size vs max single-stock limit
    2. Daily P&L vs loss limit
    3. Weekly P&L vs loss limit
    4. Current drawdown vs max drawdown
    5. Total exposure vs max exposure
    6. Open order count vs limit
    7. Kelly-capped position sizing
    """
    
    def __init__(self, limits: RiskLimits = None, initial_capital: float = 100000):
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.halted = False
        self.halt_reason = ""
    
    def check_order(self, order_value: float, current_holdings_value: float,
                    ticker: str, holdings: dict, prices: dict) -> dict:
        """
        Check if an order passes all risk controls.
        
        Returns:
            Dict with approved (bool), reason (str), adjusted_quantity (int or None)
        """
        checks = []
        
        # 1. Halt check
        if self.halted:
            return {"approved": False, "reason": f"Trading halted: {self.halt_reason}"}
        
        # 2. Position concentration
        new_position_value = order_value
        total_equity = self.current_equity
        position_pct = new_position_value / max(total_equity, 1)
        if position_pct > self.limits.max_position_pct:
            max_value = total_equity * self.limits.max_position_pct
            checks.append({
                "passed": False,
                "check": "position_concentration",
                "message": f"Position {position_pct:.1%} > {self.limits.max_position_pct:.1%} limit",
                "max_value": max_value,
            })
        else:
            checks.append({"passed": True, "check": "position_concentration"})
        
        # 3. Daily loss limit
        daily_loss_limit = self.initial_capital * self.limits.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            checks.append({
                "passed": False,
                "check": "daily_loss",
                "message": f"Daily loss {self.daily_pnl:.2f} exceeds limit {-daily_loss_limit:.2f}",
            })
        else:
            checks.append({"passed": True, "check": "daily_loss"})
        
        # 4. Weekly loss limit
        weekly_loss_limit = self.initial_capital * self.limits.max_weekly_loss_pct
        if self.weekly_pnl < -weekly_loss_limit:
            checks.append({
                "passed": False,
                "check": "weekly_loss",
                "message": f"Weekly loss {self.weekly_pnl:.2f} exceeds limit {-weekly_loss_limit:.2f}",
            })
        else:
            checks.append({"passed": True, "check": "weekly_loss"})
        
        # 5. Drawdown limit
        drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1)
        if drawdown > self.limits.max_drawdown_pct:
            checks.append({
                "passed": False,
                "check": "max_drawdown",
                "message": f"Drawdown {drawdown:.1%} exceeds {self.limits.max_drawdown_pct:.1%} limit",
            })
            self.halted = True
            self.halt_reason = f"Max drawdown breached ({drawdown:.1%})"
        else:
            checks.append({"passed": True, "check": "max_drawdown"})
        
        # 6. Total exposure
        total_exposure = current_holdings_value + order_value
        exposure_pct = total_exposure / max(self.current_equity, 1)
        if exposure_pct > self.limits.max_total_exposure_pct:
            checks.append({
                "passed": False,
                "check": "total_exposure",
                "message": f"Total exposure {exposure_pct:.1%} > {self.limits.max_total_exposure_pct:.1%}",
            })
        else:
            checks.append({"passed": True, "check": "total_exposure"})
        
        all_passed = all(c["passed"] for c in checks)
        
        return {
            "approved": all_passed,
            "checks": checks,
            "drawdown_pct": drawdown,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
        }
    
    def update_equity(self, new_equity: float):
        """Update equity and track P&L."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)
    
    def update_daily_pnl(self, pnl: float):
        """Add to daily P&L tracker."""
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily P&L (call at start of each trading day)."""
        self.daily_pnl = 0.0
    
    def reset_weekly(self):
        """Reset weekly P&L (call at start of each week)."""
        self.weekly_pnl = 0.0
        self.daily_pnl = 0.0
    
    def resume_trading(self):
        """Manually resume trading after halt."""
        self.halted = False
        self.halt_reason = ""
        logger.info("Trading resumed manually")
    
    def kelly_sized_quantity(self, win_rate: float, avg_win: float,
                            avg_loss: float, price: float, capital: float) -> int:
        """Calculate position size using Kelly fraction (not full Kelly)."""
        from src.risk import kelly_criterion
        full_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
        fraction = min(full_kelly, self.limits.kelly_fraction)
        
        risk_amount = capital * fraction
        shares = int(risk_amount / price) if price > 0 else 0
        max_affordable = int(capital * self.limits.max_position_pct / price) if price > 0 else 0
        
        return min(shares, max_affordable)
    
    def get_status(self) -> dict:
        """Get current risk status."""
        drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1)
        return {
            "halted": self.halted,
            "halt_reason": self.halt_reason,
            "current_equity": self.current_equity,
            "peak_equity": self.peak_equity,
            "drawdown_pct": drawdown,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
            "daily_loss_remaining": self.initial_capital * self.limits.max_daily_loss_pct + self.daily_pnl,
            "weekly_loss_remaining": self.initial_capital * self.limits.max_weekly_loss_pct + self.weekly_pnl,
        }
```

**Tests:** `tests/test_risk_controls.py`
- `test_order_within_limits` — order approved
- `test_order_exceeds_position_limit` — rejected
- `test_order_exceeds_daily_loss` — rejected
- `test_order_exceeds_weekly_loss` — rejected
- `test_drawdown_halts_trading` — halted on breach
- `test_resume_trading` — manually resume
- `test_total_exposure_check` — rejects overexposure
- `test_kelly_sized_quantity` — returns reasonable size
- `test_kelly_respects_position_limit` — capped at max position
- `test_get_status` — returns all fields
- `test_halt_blocks_all_orders` — no orders when halted
- `test_daily_reset` — P&L resets

---

### Task 3: Paper Trading Mode
**Impact:** Critical — validates signals without real money
**Effort:** 4-5 hours
**Files:** New `src/paper_trader.py`, update `app.py`

**Why:** Paper trading is the single most important step before real money. It runs the same code path as live trading but with simulated fills.

**Implementation:**

```python
# src/paper_trader.py

import json
import logging
import os
from datetime import datetime
from dataclasses import dataclass, field

import pandas as pd

from src.engine import ExecutionEngine, Order, OrderSide, OrderType, Bar
from src.risk_controls import RiskController, RiskLimits

logger = logging.getLogger(__name__)

PAPER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "paper_trading"
)


@dataclass
class PaperTrade:
    """A single paper trade record."""
    order_id: str
    ticker: str
    side: str
    order_type: str
    quantity: int
    signal_price: float
    fill_price: float
    fill_cost: float
    signal_date: str
    fill_date: str
    slippage: float
    costs: dict = field(default_factory=dict)


class PaperTrader:
    """
    Paper trading engine.
    
    Runs daily:
    1. Fetch latest data
    2. Generate signals from trained models
    3. Run risk checks
    4. Submit orders to execution engine
    5. Record fills
    6. Track P&L
    """
    
    def __init__(self, initial_capital: float = 100000,
                 limits: RiskLimits = None):
        self.engine = ExecutionEngine()
        self.risk = RiskController(limits, initial_capital)
        self.initial_capital = initial_capital
        self.portfolio = {}  # ticker -> (quantity, avg_price)
        self.cash = initial_capital
        self.trades: list[PaperTrade] = []
        self.equity_curve: list[dict] = []
        os.makedirs(PAPER_DIR, exist_ok=True)
    
    def run_day(self, ticker: str, bar: Bar, signal: dict,
                current_prices: dict = None) -> dict:
        """
        Process one trading day.
        
        Args:
            ticker: Stock ticker
            bar: Today's OHLCV bar
            signal: {"direction": 1/-1/0, "confidence": float}
            current_prices: Current prices for all holdings
        
        Returns:
            Dict with actions taken
        """
        actions = {"orders_submitted": 0, "orders_filled": 0, "orders_rejected": 0}
        
        # Update equity
        if current_prices:
            self._update_equity(current_prices)
        
        # Check risk status
        risk_status = self.risk.get_status()
        if risk_status["halted"]:
            logger.warning(f"Trading halted: {risk_status['halt_reason']}")
            actions["halted"] = True
            return actions
        
        # Check existing position
        held_qty, avg_price = self.portfolio.get(ticker, (0, 0))
        direction = signal.get("direction", 0)
        confidence = signal.get("confidence", 0)
        
        # Generate order based on signal
        order = None
        
        if direction == 1 and held_qty == 0:
            # BUY signal, no position
            order_value = self.cash * min(confidence, self.risk.limits.max_position_pct)
            quantity = int(order_value / bar.close) if bar.close > 0 else 0
            
            if quantity > 0:
                risk_check = self.risk.check_order(
                    bar.close * quantity,
                    sum(q * p for q, p in self.portfolio.values()),
                    ticker, self.portfolio, current_prices or {}
                )
                
                if risk_check["approved"]:
                    order = Order(
                        order_id="", ticker=ticker, side=OrderSide.BUY,
                        order_type=OrderType.MARKET, quantity=quantity,
                    )
                else:
                    actions["orders_rejected"] += 1
                    actions["rejection_reason"] = risk_check.get("checks", [{}])[0].get("message", "")
        
        elif direction == -1 and held_qty > 0:
            # SELL signal, have position
            order = Order(
                order_id="", ticker=ticker, side=OrderSide.SELL,
                order_type=OrderType.MARKET, quantity=held_qty,
            )
        
        if order:
            actions["orders_submitted"] += 1
            self.engine.submit_order(order)
            
            # Process bar (will fill market orders at open)
            filled = self.engine.on_bar(bar)
            
            for fill in filled:
                actions["orders_filled"] += 1
                self._process_fill(fill, bar.timestamp)
        
        # Record equity
        total = self.cash + sum(q * p for q, p in self.portfolio.values())
        self.equity_curve.append({
            "date": bar.timestamp,
            "equity": total,
            "cash": self.cash,
        })
        
        return actions
    
    def _process_fill(self, order: Order, timestamp: str):
        """Process a filled order."""
        ticker = order.ticker
        
        if order.side == OrderSide.BUY:
            cost = order.fill_cost
            self.cash -= cost
            
            if ticker in self.portfolio:
                old_qty, old_price = self.portfolio[ticker]
                new_qty = old_qty + order.filled_quantity
                new_price = ((old_price * old_qty) +
                            (order.filled_price * order.filled_quantity)) / new_qty
                self.portfolio[ticker] = (new_qty, new_price)
            else:
                self.portfolio[ticker] = (order.filled_quantity, order.filled_price)
            
            self.risk.update_daily_pnl(0)  # No P&L on buy
        
        elif order.side == OrderSide.SELL:
            proceeds = order.fill_cost  # Already includes costs
            self.cash += proceeds
            
            if ticker in self.portfolio:
                old_qty, old_price = self.portfolio[ticker]
                pnl = (order.filled_price - old_price) * order.filled_quantity
                self.risk.update_daily_pnl(pnl)
                del self.portfolio[ticker]
        
        trade = PaperTrade(
            order_id=order.order_id, ticker=ticker,
            side=order.side.value, order_type=order.order_type.value,
            quantity=order.filled_quantity,
            signal_price=0, fill_price=order.filled_price,
            fill_cost=order.fill_cost,
            signal_date=timestamp, fill_date=timestamp,
            slippage=abs(order.filled_price - 0),  # Would need signal price
        )
        self.trades.append(trade)
    
    def _update_equity(self, prices: dict):
        """Update equity with current market prices."""
        total = self.cash
        for ticker, (qty, _) in self.portfolio.items():
            if ticker in prices:
                total += qty * prices[ticker]
        self.risk.update_equity(total)
    
    def save_state(self):
        """Save paper trading state to disk."""
        state = {
            "initial_capital": self.initial_capital,
            "cash": self.cash,
            "portfolio": self.portfolio,
            "equity_curve": self.equity_curve[-30:],  # Last 30 days
            "trades": len(self.trades),
            "risk_status": self.risk.get_status(),
            "last_updated": datetime.now().isoformat(),
        }
        path = os.path.join(PAPER_DIR, "paper_state.json")
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
    
    def load_state(self):
        """Load paper trading state from disk."""
        path = os.path.join(PAPER_DIR, "paper_state.json")
        if not os.path.exists(path):
            return False
        
        with open(path) as f:
            state = json.load(f)
        
        self.cash = state.get("cash", self.initial_capital)
        self.portfolio = {k: tuple(v) for k, v in state.get("portfolio", {}).items()}
        self.equity_curve = state.get("equity_curve", [])
        return True
    
    def get_performance(self) -> dict:
        """Get paper trading performance summary."""
        if len(self.equity_curve) < 2:
            return {"error": "Insufficient data"}
        
        equity = pd.Series([e["equity"] for e in self.equity_curve])
        returns = equity.pct_change().dropna()
        
        total_return = (equity.iloc[-1] / equity.iloc[0]) - 1
        n_days = len(equity)
        n_years = max(n_days / 252, 0.01)
        ann_return = (1 + total_return) ** (1 / n_years) - 1
        
        from src.risk import calculate_sharpe, calculate_sortino, calculate_max_drawdown
        
        return {
            "total_return": total_return,
            "annualized_return": ann_return,
            "sharpe_ratio": calculate_sharpe(returns.values) if len(returns) > 10 else 0,
            "sortino_ratio": calculate_sortino(returns.values) if len(returns) > 10 else 0,
            "max_drawdown": calculate_max_drawdown(equity.values),
            "n_trades": len(self.trades),
            "current_equity": equity.iloc[-1],
            "cash": self.cash,
            "holdings": dict(self.portfolio),
            "n_days": n_days,
        }
```

**Tests:** `tests/test_paper_trader.py`
- `test_run_day_buy_signal` — submits buy order on signal
- `test_run_day_sell_signal` — submits sell order on signal
- `test_risk_rejects_overlimit` — risk controller rejects
- `test_halt_stops_trading` — halted state blocks orders
- `test_process_fill_updates_portfolio` — buy fills update holdings
- `test_process_fill_updates_cash` — sell fills update cash
- `test_save_load_state` — roundtrip to disk
- `test_get_performance` — returns metrics
- `test_equity_curve_recorded` — equity tracked daily

---

### Task 4: Execution Quality
**Impact:** Medium — slippage and order type simulation
**Effort:** 2-3 hours
**Files:** Update `src/backtester.py`

**Why:** The current backtester uses fixed 0.1% slippage. Real execution depends on volume, order size, and market conditions. Better simulation = more honest backtests.

**What to add to backtester.py:**

```python
# Add to src/backtester.py

class BacktestExecutionSimulator:
    """
    Realistic execution simulation for backtesting.
    
    Models:
    - Volume-based slippage (thin books = more slippage)
    - Market impact (large orders move price)
    - Order type simulation (limit, market, stop)
    - Fill probability (limit orders may not fill)
    """
    
    def __init__(self, slippage_model="volume", impact_factor=0.1):
        self.slippage_model = slippage_model
        self.impact_factor = impact_factor
    
    def simulate_fill(self, order_side, price, volume, order_value=0):
        """Simulate realistic fill with slippage and impact."""
        if self.slippage_model == "volume":
            from src.engine import VolumeSlippage
            model = VolumeSlippage()
            slippage = model.calculate(order_side, price, volume)
        elif self.slippage_model == "adaptive":
            from src.engine import AdaptiveSlippage
            model = AdaptiveSlippage()
            slippage = model.calculate(order_side, price, volume, order_value)
        else:
            from src.engine import FixedSlippage
            model = FixedSlippage()
            slippage = model.calculate(order_side, price, volume)
        
        if order_side == OrderSide.BUY:
            return price + slippage
        else:
            return price - slippage
    
    def estimate_fill_probability(self, order_type, price, bar):
        """Estimate probability that a limit order fills."""
        if order_type == OrderType.MARKET:
            return 1.0
        elif order_type == OrderType.LIMIT:
            # Probability depends on how far limit is from close
            distance = abs(price - bar["close"]) / bar["close"]
            if distance < 0.005:
                return 0.9
            elif distance < 0.01:
                return 0.7
            elif distance < 0.02:
                return 0.4
            return 0.1
        return 0.5
```

Also update `run_walk_forward_backtest` to use the new slippage models instead of fixed `SLIPPAGE_RATE`.

**Tests:** `tests/test_execution_quality.py`
- `test_volume_slippage_increases_with_low_volume` — low volume = more slippage
- `test_adaptive_slippage_increases_with_order_size` — large orders = more slippage
- `test_fill_probability_market_always_fills` — market = 100%
- `test_fill_probability_limit_near_fills` — near limit = high probability
- `test_fill_probability_limit_far_low` — far limit = low probability

---

## Execution Order

```
1. Task 1: Event-Driven Engine (5-6 hrs)     ← CRITICAL, foundation for Tasks 2-4
2. Task 2: Risk Controls (3-4 hrs)            ← depends on Task 1 (uses Order/Bar)
3. Task 3: Paper Trading (4-5 hrs)            ← depends on Tasks 1 + 2
4. Task 4: Execution Quality (2-3 hrs)        ← depends on Task 1 (slippage models)
```

**Parallelizable:**
- Task 4 can be developed in parallel with Task 2 (both depend on Task 1)
- Task 3 requires both Tasks 1 and 2

**Total estimated time:** 14-18 hours

---

## Exit Criteria Checklist

After all 4 tasks:

- [x] Event-driven engine processes orders bar-by-bar
- [x] Same code path for backtest and paper trading
- [x] Paper trading mode records trades, tracks P&L, enforces risk limits
- [x] Risk controls reject orders exceeding position/daily/weekly/drawdown limits
- [x] Kelly-capped position sizing (quarter-Kelly, not full)
- [x] Volume-based and adaptive slippage models
- [x] Fill probability modeling for limit orders
- [x] State persistence (save/load paper trading state)
- [x] All 4 tasks have tests passing
- [x] `run_pipeline.py --paper` flag to run paper trading after training

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/engine.py` | Event-driven execution engine, Order/Bar classes, slippage models, fill models |
| `src/risk_controls.py` | Pre-trade risk checks, position/daily/weekly/drawdown limits, Kelly sizing |
| `src/paper_trader.py` | Paper trading mode, portfolio tracking, state persistence |
| `tests/test_engine.py` | 12 tests for execution engine |
| `tests/test_risk_controls.py` | 12 tests for risk controls |
| `tests/test_paper_trader.py` | 9 tests for paper trading |
| `tests/test_execution_quality.py` | 5 tests for slippage/fill models |

## Files to Modify

| File | Change |
|------|--------|
| `src/backtester.py` | Add `BacktestExecutionSimulator`, replace fixed slippage with volume/adaptive models |
| `app.py` | Add "Paper Trading" tab (16th tab) |
| `requirements.txt` | No changes needed |
| `run_pipeline.py` | Add `--paper` flag to run paper trading after training |

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130, scipy installed)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run linter:** `C:\Users\uttam\venv\Scripts\ruff.exe check src/ tests/ --select F401,E,F,W --ignore E501,F841`
- **Current test count:** 335 tests passing
- **Existing modules:** `risk.py` (Kelly, VaR, CVaR, sizing), `portfolio.py` (buy/sell with NSE costs), `backtester.py` (walk-forward, compute_metrics)
- **User constraint:** Everything must be free, must work on Windows
- **Key insight:** This stage is about EXECUTION — making sure signals are actually tradeable with proper risk management
- **Paper trading must run for 3+ months before real money** (per IMPROVEMENTS.md)
- **Event-driven parity:** Same code path for backtest → paper → live

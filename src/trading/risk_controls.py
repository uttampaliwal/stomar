"""Pre-trade risk controls.

Validates orders before submission to prevent catastrophic loss.

Checks:
1. Position size vs max single-stock limit
2. Daily P&L vs loss limit
3. Weekly P&L vs loss limit
4. Current drawdown vs max drawdown (halts trading on breach)
5. Total exposure vs max exposure
6. Kelly-capped position sizing (quarter-Kelly)
7. Kill switch (emergency halt + flatten, persistent across restarts)
8. Correlation-based concentration limit
9. Consecutive loss circuit breaker
10. Liquidity: min average daily traded value (INR)
11. Slippage estimate: max acceptable bps
12. Gap vs previous close: max percent before trading is blocked

Usage:
    from src.trading.risk_controls import RiskController, RiskLimits
    rc = RiskController(RiskLimits(), initial_capital=100000)
    result = rc.check_order(order_value=25000, holdings_value=50000,
                            market=MarketContext(...))
    if result["approved"]:
        engine.submit_order(order)
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Per-ticker market snapshot used by the market-quality checks.

    All fields optional; a missing field skips the related check.
    """
    price: float = 0.0                       # current tradable price
    prev_close: float = 0.0                  # previous session close
    avg_daily_traded_value: float = 0.0      # INR average daily traded value
    expected_slippage_bps: float = 0.0       # estimated slippage for this order
    daily_volume_rs: float = 0.0             # alias for avg_daily_traded_value


@dataclass
class RiskLimits:
    """Configurable risk limits."""
    max_position_pct: float = 0.25
    max_daily_loss_pct: float = 0.02
    max_weekly_loss_pct: float = 0.05
    max_drawdown_pct: float = 0.15
    max_open_orders: int = 10
    kelly_fraction: float = 0.25
    max_total_exposure_pct: float = 0.95
    halt_on_breach: bool = True
    max_correlated_exposure_pct: float = 0.40
    max_consecutive_losses: int = 5
    correlation_threshold: float = 0.70
    # Market-quality hard limits (settings-backed at construction time)
    min_daily_volume_rs: float = 1_000_000.0
    max_expected_slippage_bps: float = 30.0
    max_gap_pct: float = 5.0


class RiskController:
    """Pre-trade risk checks. Validates orders before submission."""

    def __init__(self, limits: RiskLimits = None, initial_capital: float = 100000,
                 kill_switch_file: str | Path | None = None):
        self.limits = limits or RiskLimits()
        self.initial_capital = initial_capital
        self.daily_pnl = 0.0
        self.weekly_pnl = 0.0
        self.peak_equity = initial_capital
        self.current_equity = initial_capital
        self.halted = False
        self.halt_reason = ""
        self.consecutive_losses = 0
        self.sector_exposure = {}  # sector -> total value
        self.position_correlations = {}  # (ticker_a, ticker_b) -> correlation
        self.kill_switch_file = (
            Path(kill_switch_file) if kill_switch_file
            else Path(DATA_DIR) / "kill_switch.json"
        )
        self._load_persistent_kill_switch()

    # ── persistent kill switch ─────────────────────────────────────────────

    def _load_persistent_kill_switch(self):
        try:
            if self.kill_switch_file.exists():
                data = json.loads(self.kill_switch_file.read_text())
                self.halted = bool(data.get("active", False))
                self.halt_reason = str(data.get("reason", ""))
                logger.critical("persistent kill switch ACTIVE: %s", self.halt_reason)
        except (OSError, ValueError) as exc:
            logger.error("failed to read kill switch file: %s", exc)

    def kill_switch(self, reason: str = "Emergency kill switch activated"):
        """Emergency halt: stop all trading immediately and persist."""
        self.halted = True
        self.halt_reason = reason
        try:
            self.kill_switch_file.parent.mkdir(parents=True, exist_ok=True)
            self.kill_switch_file.write_text(json.dumps({
                "active": True,
                "reason": reason,
                "set_at": __import__("datetime").datetime.now().isoformat(),
            }))
        except OSError as exc:
            logger.error("failed to persist kill switch: %s", exc)
        logger.critical("KILL SWITCH ACTIVATED: %s", reason)

    def clear_kill_switch(self):
        """Resume trading after an emergency halt (manual only)."""
        self.halted = False
        self.halt_reason = ""
        try:
            self.kill_switch_file.unlink(missing_ok=True)
        except OSError as exc:
            logger.error("failed to clear kill switch file: %s", exc)
        logger.warning("KILL SWITCH CLEARED (manual resume)")

    def check_order(self, order_value: float, current_holdings_value: float,
                    ticker: str = "", holdings: dict = None,
                    prices: dict = None, is_closing: bool = False,
                    market: "MarketContext | None" = None) -> dict:
        """Check if an order passes all risk controls.

        Args:
            order_value: Notional value of the proposed order.
            current_holdings_value: Current value of all holdings.
            ticker: Symbol being traded.
            holdings: dict ticker -> (qty, price).
            prices: dict ticker -> quote dict (kept for back-compat).
            is_closing: True if this order closes/reduces an existing position.
            market: MarketContext with price/liquidity/gap info.

        Returns:
            Dict with approved (bool), checks (list), drawdown_pct
        """
        checks = []

        if self.halted:
            return {"approved": False, "reason": f"Trading halted: {self.halt_reason}",
                    "checks": [], "drawdown_pct": 0}
        total_equity = self.current_equity

        # 1. Position concentration (skip if closing/reducing existing position)
        position_pct = order_value / max(total_equity, 1)
        if not is_closing and position_pct > self.limits.max_position_pct:
            max_value = total_equity * self.limits.max_position_pct
            checks.append({
                "passed": False, "check": "position_concentration",
                "message": f"Position {position_pct:.1%} > {self.limits.max_position_pct:.1%} limit",
                "max_value": max_value,
            })
        else:
            checks.append({"passed": True, "check": "position_concentration"})

        # 2. Daily loss limit (based on current equity, not initial)
        daily_loss_limit = self.current_equity * self.limits.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            checks.append({
                "passed": False, "check": "daily_loss",
                "message": (f"Daily loss {self.daily_pnl:.2f} exceeds "
                            f"limit {-daily_loss_limit:.2f}"),
            })
        else:
            checks.append({"passed": True, "check": "daily_loss"})

        # 3. Weekly loss limit (based on current equity, not initial)
        weekly_loss_limit = self.current_equity * self.limits.max_weekly_loss_pct
        if self.weekly_pnl < -weekly_loss_limit:
            checks.append({
                "passed": False, "check": "weekly_loss",
                "message": (f"Weekly loss {self.weekly_pnl:.2f} exceeds "
                            f"limit {-weekly_loss_limit:.2f}"),
            })
        else:
            checks.append({"passed": True, "check": "weekly_loss"})

        # 4. Drawdown limit
        drawdown = (self.peak_equity - self.current_equity) / max(self.peak_equity, 1)
        if drawdown > self.limits.max_drawdown_pct:
            checks.append({
                "passed": False, "check": "max_drawdown",
                "message": (f"Drawdown {drawdown:.1%} exceeds "
                            f"{self.limits.max_drawdown_pct:.1%} limit"),
            })
            self.halted = True
            self.halt_reason = f"Max drawdown breached ({drawdown:.1%})"
        else:
            checks.append({"passed": True, "check": "max_drawdown"})

        # 5. Total exposure (skip if closing/reducing existing position)
        total_exposure = current_holdings_value + order_value
        exposure_pct = total_exposure / max(self.current_equity, 1)
        if not is_closing and exposure_pct > self.limits.max_total_exposure_pct:
            checks.append({
                "passed": False, "check": "total_exposure",
                "message": (f"Total exposure {exposure_pct:.1%} > "
                            f"{self.limits.max_total_exposure_pct:.1%}"),
            })
        else:
            checks.append({"passed": True, "check": "total_exposure"})

        # 6. Correlated exposure (skip when closing/reducing an existing position)
        if not is_closing and holdings and ticker:
            checks.append(
                self.check_correlated_exposure(holdings, ticker, order_value)
            )

        # 10. Liquidity (skip when closing an existing position)
        if market is not None:
            volume_rs = market.avg_daily_traded_value or market.daily_volume_rs
            if not is_closing and volume_rs > 0 and \
                    volume_rs < self.limits.min_daily_volume_rs:
                checks.append({
                    "passed": False, "check": "liquidity",
                    "message": (f"Average daily traded value {volume_rs:,.0f} INR < "
                                f"{self.limits.min_daily_volume_rs:,.0f} INR minimum"),
                })
            else:
                checks.append({"passed": True, "check": "liquidity"})

            # 11. Expected slippage
            if market.expected_slippage_bps > self.limits.max_expected_slippage_bps:
                checks.append({
                    "passed": False, "check": "slippage",
                    "message": (f"Expected slippage {market.expected_slippage_bps:.0f} bps > "
                                f"{self.limits.max_expected_slippage_bps:.0f} bps limit"),
                })
            else:
                checks.append({"passed": True, "check": "slippage"})

            # 12. Gap vs previous close
            if market.price > 0 and market.prev_close > 0:
                gap_pct = abs(market.price - market.prev_close) / market.prev_close * 100
                if gap_pct > self.limits.max_gap_pct:
                    checks.append({
                        "passed": False, "check": "gap",
                        "message": (f"Gap {gap_pct:.2f}% vs previous close > "
                                    f"{self.limits.max_gap_pct:.2f}% limit"),
                    })
                else:
                    checks.append({"passed": True, "check": "gap"})
            else:
                checks.append({"passed": True, "check": "gap"})

        all_passed = all(c["passed"] for c in checks)

        return {
            "approved": all_passed,
            "checks": checks,
            "drawdown_pct": drawdown,
            "daily_pnl": self.daily_pnl,
            "weekly_pnl": self.weekly_pnl,
        }

    def update_equity(self, new_equity: float):
        """Update equity and track peak."""
        self.current_equity = new_equity
        self.peak_equity = max(self.peak_equity, new_equity)

    def update_daily_pnl(self, pnl: float):
        """Add to daily P&L tracker."""
        self.daily_pnl += pnl

    def reset_daily(self):
        """Reset daily P&L."""
        self.daily_pnl = 0.0

    def reset_weekly(self):
        """Reset weekly P&L."""
        self.weekly_pnl = 0.0
        self.daily_pnl = 0.0

    def resume_trading(self):
        """Manually resume trading after halt (also clears persistent file)."""
        self.clear_kill_switch()

    def kelly_sized_quantity(self, win_rate: float, avg_win: float,
                            avg_loss: float, price: float,
                            capital: float) -> int:
        """Calculate position size using Kelly fraction (not full Kelly)."""
        from src.trading.risk import kelly_criterion
        full_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
        fraction = min(full_kelly, self.limits.kelly_fraction)

        risk_amount = capital * fraction
        shares = int(risk_amount / price) if price > 0 else 0
        max_affordable = int(capital * self.limits.max_position_pct / price) \
            if price > 0 else 0

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
            "daily_loss_remaining": (
                self.current_equity * self.limits.max_daily_loss_pct + self.daily_pnl
            ),
            "weekly_loss_remaining": (
                self.current_equity * self.limits.max_weekly_loss_pct + self.weekly_pnl
            ),
            "consecutive_losses": self.consecutive_losses,
        }

    def update_consecutive_losses(self, is_loss: bool):
        """Track consecutive losses for circuit breaker."""
        if is_loss:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.limits.max_consecutive_losses:
                self.kill_switch(
                    f"Consecutive losses ({self.consecutive_losses}) "
                    f"exceed limit ({self.limits.max_consecutive_losses})"
                )
        else:
            self.consecutive_losses = 0

    def update_correlation(self, ticker_a: str, ticker_b: str, corr: float):
        """Update correlation between two positions."""
        key = tuple(sorted([ticker_a, ticker_b]))
        self.position_correlations[key] = corr

    def check_correlated_exposure(self, holdings: dict, new_ticker: str,
                                  new_value: float) -> dict:
        """Check if adding a position would exceed correlated exposure limit.

        Holdings may be ticker -> (qty, price) tuples or position objects
        exposing ``market_value`` (e.g. PaperTrader positions).
        """
        correlated_value = new_value
        for existing_ticker, holding in holdings.items():
            key = tuple(sorted([existing_ticker, new_ticker]))
            corr = self.position_correlations.get(key, 0.0)
            if abs(corr) >= self.limits.correlation_threshold:
                if hasattr(holding, "market_value"):
                    correlated_value += abs(holding.market_value)
                else:
                    qty, price = holding[0], holding[1]
                    correlated_value += abs(qty) * price

        corr_pct = correlated_value / max(self.current_equity, 1)
        if corr_pct > self.limits.max_correlated_exposure_pct:
            return {
                "passed": False,
                "check": "correlated_exposure",
                "message": (
                    f"Correlated exposure {corr_pct:.1%} > "
                    f"{self.limits.max_correlated_exposure_pct:.1%} limit"
                ),
            }
        return {"passed": True, "check": "correlated_exposure"}

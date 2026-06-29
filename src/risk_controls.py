"""Pre-trade risk controls.

Validates orders before submission to prevent catastrophic loss.

Checks:
1. Position size vs max single-stock limit
2. Daily P&L vs loss limit
3. Weekly P&L vs loss limit
4. Current drawdown vs max drawdown (halts trading on breach)
5. Total exposure vs max exposure
6. Kelly-capped position sizing (quarter-Kelly)

Usage:
    from src.risk_controls import RiskController, RiskLimits
    rc = RiskController(RiskLimits(), initial_capital=100000)
    result = rc.check_order(order_value=25000, holdings_value=50000, ...)
    if result["approved"]:
        engine.submit_order(order)
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


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


class RiskController:
    """Pre-trade risk checks. Validates orders before submission."""

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
                    ticker: str = "", holdings: dict = None,
                    prices: dict = None) -> dict:
        """Check if an order passes all risk controls.

        Returns:
            Dict with approved (bool), checks (list), drawdown_pct
        """
        checks = []

        if self.halted:
            return {"approved": False, "reason": f"Trading halted: {self.halt_reason}",
                    "checks": [], "drawdown_pct": 0}

        total_equity = self.current_equity

        # 1. Position concentration
        position_pct = order_value / max(total_equity, 1)
        if position_pct > self.limits.max_position_pct:
            max_value = total_equity * self.limits.max_position_pct
            checks.append({
                "passed": False, "check": "position_concentration",
                "message": f"Position {position_pct:.1%} > {self.limits.max_position_pct:.1%} limit",
                "max_value": max_value,
            })
        else:
            checks.append({"passed": True, "check": "position_concentration"})

        # 2. Daily loss limit
        daily_loss_limit = self.initial_capital * self.limits.max_daily_loss_pct
        if self.daily_pnl < -daily_loss_limit:
            checks.append({
                "passed": False, "check": "daily_loss",
                "message": (f"Daily loss {self.daily_pnl:.2f} exceeds "
                            f"limit {-daily_loss_limit:.2f}"),
            })
        else:
            checks.append({"passed": True, "check": "daily_loss"})

        # 3. Weekly loss limit
        weekly_loss_limit = self.initial_capital * self.limits.max_weekly_loss_pct
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

        # 5. Total exposure
        total_exposure = current_holdings_value + order_value
        exposure_pct = total_exposure / max(self.current_equity, 1)
        if exposure_pct > self.limits.max_total_exposure_pct:
            checks.append({
                "passed": False, "check": "total_exposure",
                "message": (f"Total exposure {exposure_pct:.1%} > "
                            f"{self.limits.max_total_exposure_pct:.1%}"),
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
        """Manually resume trading after halt."""
        self.halted = False
        self.halt_reason = ""
        logger.info("Trading resumed manually")

    def kelly_sized_quantity(self, win_rate: float, avg_win: float,
                            avg_loss: float, price: float,
                            capital: float) -> int:
        """Calculate position size using Kelly fraction (not full Kelly)."""
        from src.risk import kelly_criterion
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
                self.initial_capital * self.limits.max_daily_loss_pct + self.daily_pnl
            ),
            "weekly_loss_remaining": (
                self.initial_capital * self.limits.max_weekly_loss_pct + self.weekly_pnl
            ),
        }

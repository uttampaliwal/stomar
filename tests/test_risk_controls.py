"""Tests for src/risk_controls.py."""

import pytest

from src.risk_controls import RiskController


def _rc(**kwargs):
    defaults = dict(initial_capital=100_000)
    defaults.update(kwargs)
    return RiskController(**defaults)


# ── Position Concentration ──

class TestPositionConcentration:
    def test_approved_within_limit(self):
        rc = _rc()
        result = rc.check_order(order_value=20_000, current_holdings_value=0)
        assert result["approved"] is True

    def test_rejected_over_limit(self):
        rc = _rc()
        result = rc.check_order(order_value=30_000, current_holdings_value=0)
        assert result["approved"] is False
        checks = {c["check"]: c["passed"] for c in result["checks"]}
        assert checks["position_concentration"] is False

    def test_exact_limit_passes(self):
        rc = _rc()
        result = rc.check_order(order_value=25_000, current_holdings_value=0)
        assert result["approved"] is True


# ── Daily Loss Limit ──

class TestDailyLossLimit:
    def test_within_limit(self):
        rc = _rc()
        rc.daily_pnl = -500
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is True

    def test_exceeded(self):
        rc = _rc()
        rc.daily_pnl = -3_000
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is False
        checks = {c["check"]: c["passed"] for c in result["checks"]}
        assert checks["daily_loss"] is False


# ── Weekly Loss Limit ──

class TestWeeklyLossLimit:
    def test_within_limit(self):
        rc = _rc()
        rc.weekly_pnl = -2_000
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is True

    def test_exceeded(self):
        rc = _rc()
        rc.weekly_pnl = -6_000
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is False
        checks = {c["check"]: c["passed"] for c in result["checks"]}
        assert checks["weekly_loss"] is False


# ── Drawdown Limit ──

class TestDrawdownLimit:
    def test_no_drawdown(self):
        rc = _rc(initial_capital=100_000)
        rc.update_equity(100_000)
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is True
        assert result["drawdown_pct"] == 0.0

    def test_moderate_drawdown(self):
        rc = _rc(initial_capital=100_000)
        rc.update_equity(90_000)
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is True
        assert result["drawdown_pct"] == pytest.approx(0.1)

    def test_breach_halts_trading(self):
        rc = _rc(initial_capital=100_000)
        rc.update_equity(84_000)
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is False
        assert rc.halted is True

    def test_halt_blocks_future_orders(self):
        rc = _rc(initial_capital=100_000)
        rc.halted = True
        result = rc.check_order(order_value=1_000, current_holdings_value=0)
        assert result["approved"] is False
        assert "halted" in result["reason"].lower()


# ── Total Exposure ──

class TestTotalExposure:
    def test_within_limit(self):
        rc = _rc()
        result = rc.check_order(order_value=10_000, current_holdings_value=60_000)
        assert result["approved"] is True

    def test_exceeds_limit(self):
        rc = _rc()
        result = rc.check_order(order_value=70_000, current_holdings_value=30_000)
        assert result["approved"] is False
        checks = {c["check"]: c["passed"] for c in result["checks"]}
        assert checks["total_exposure"] is False


# ── Resume Trading ──

class TestResume:
    def test_resume_after_halt(self):
        rc = _rc()
        rc.halted = True
        rc.halt_reason = "Max drawdown breached"
        rc.resume_trading()
        assert rc.halted is False
        assert rc.halt_reason == ""
        result = rc.check_order(order_value=10_000, current_holdings_value=0)
        assert result["approved"] is True


# ── Kelly Sizing ──

class TestKellySizing:
    def test_positive_edge_returns_positive_quantity(self):
        rc = _rc()
        qty = rc.kelly_sized_quantity(
            win_rate=0.6, avg_win=2000, avg_loss=1000, price=500, capital=100_000
        )
        assert qty > 0

    def test_zero_edge_returns_zero(self):
        rc = _rc()
        qty = rc.kelly_sized_quantity(
            win_rate=0.5, avg_win=1000, avg_loss=1000, price=500, capital=100_000
        )
        assert qty == 0

    def test_does_not_exceed_max_position(self):
        rc = _rc()
        qty = rc.kelly_sized_quantity(
            win_rate=0.8, avg_win=5000, avg_loss=1000, price=100, capital=100_000
        )
        max_value = 100_000 * 0.25
        assert qty * 100 <= max_value


# ── Status ──

class TestStatus:
    def test_status_keys(self):
        rc = _rc()
        status = rc.get_status()
        assert "halted" in status
        assert "drawdown_pct" in status
        assert "daily_pnl" in status
        assert "weekly_pnl" in status

    def test_reset_daily(self):
        rc = _rc()
        rc.daily_pnl = -1000
        rc.reset_daily()
        assert rc.daily_pnl == 0.0

    def test_reset_weekly(self):
        rc = _rc()
        rc.weekly_pnl = -3000
        rc.daily_pnl = -1000
        rc.reset_weekly()
        assert rc.weekly_pnl == 0.0
        assert rc.daily_pnl == 0.0

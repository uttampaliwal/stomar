"""Tests for the hardware circuit breakers & risk control engine."""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from services.risk_guard import (
    RiskGuard,
    RiskGuardLimits,
    compute_volatility_scalar,
    _ist_date,
)


@pytest.fixture
def guard(tmp_path):
    """Fresh guard with isolated state file and a known override token."""
    return RiskGuard(
        limits=RiskGuardLimits(),
        initial_equity=1_000_000.0,
        state_path=str(tmp_path / "risk_guard_state.json"),
        override_token="SECRET_TOKEN",
    )


class TestMaxDailyLoss:
    def test_buy_blocked_after_2pct_intraday_loss(self, guard):
        guard.update_equity(990_000)  # -1%
        assert guard.check_order(order_value=10_000, ticker="X.NS")["approved"] is True

        guard.update_equity(979_500)  # -2.05%
        result = guard.check_order(order_value=10_000, ticker="X.NS")
        assert result["approved"] is False
        assert any(c["check"] == "daily_loss" and not c["passed"] for c in result["checks"])

    def test_sell_never_blocked_by_daily_loss(self, guard):
        guard.update_equity(950_000)  # -5%
        result = guard.check_order(order_value=50_000, ticker="X.NS", is_buy=False)
        assert result["approved"] is True

    def test_daily_loss_resets_next_session(self, guard, monkeypatch):
        guard.update_equity(950_000)
        assert guard.check_order(order_value=100, ticker="X.NS")["approved"] is False

        monkeypatch.setattr("services.risk_guard._ist_date", lambda: "2099-01-02")
        guard.update_equity(960_000)
        assert guard.daily_pnl == pytest.approx(0.0)
        assert guard.check_order(order_value=100, ticker="X.NS")["approved"] is True

    def test_realized_pnl_counts_toward_loss_limit(self, guard):
        guard.update_equity(995_000)
        guard.record_realized_pnl(-20_000)
        assert guard.check_order(order_value=100, ticker="X.NS")["approved"] is False


class TestMaxDrawdownFreeze:
    def test_freezes_at_8pct_drawdown(self, guard):
        guard.update_equity(920_500)  # -7.95%
        assert guard.halted is False

        guard.update_equity(919_000)  # -8.1%
        assert guard.halted is True
        assert "drawdown" in guard.halt_reason.lower()
        assert guard.check_order(order_value=10_000, ticker="X.NS")["approved"] is False

    def test_freezed_state_blocks_all_new_buys(self, guard):
        guard.update_equity(900_000)
        result = guard.check_order(order_value=5_000, ticker="X.NS")
        assert result["approved"] is False
        assert result["halted"] is True

    def test_sells_allowed_during_freeze(self, guard):
        guard.update_equity(900_000)
        result = guard.check_order(
            order_value=50_000, ticker="X.NS", is_buy=False, is_closing=True
        )
        assert result["approved"] is True

    def test_resume_requires_correct_token(self, guard):
        guard.update_equity(900_000)
        assert guard.resume_trading("wrong")["resumed"] is False
        assert guard.halted is True
        assert guard.resume_trading("SECRET_TOKEN")["resumed"] is True
        assert guard.halted is False
        guard.update_equity(995_000)  # recover above the daily loss line
        result = guard.check_order(order_value=10_000, ticker="X.NS")
        assert result["approved"] is True

    def test_resume_without_configured_token_rejected(self, tmp_path):
        g = RiskGuard(
            limits=RiskGuardLimits(),
            initial_equity=1_000_000.0,
            state_path=str(tmp_path / "state.json"),
            override_token="",
        )
        g.update_equity(900_000)
        assert g.resume_trading("anything")["resumed"] is False

    def test_freeze_survives_restart(self, tmp_path):
        path = str(tmp_path / "state.json")
        g1 = RiskGuard(initial_equity=1_000_000.0, state_path=path, override_token="T")
        g1.update_equity(900_000)
        assert g1.halted is True

        g2 = RiskGuard(initial_equity=1_000_000.0, state_path=path, override_token="T")
        assert g2.halted is True
        assert g2.halt_reason == g1.halt_reason


class TestVolatilityScalar:
    def test_no_breach_scalar_is_one(self):
        result = compute_volatility_scalar(vix=18.0, atr_current=1.0, atr_average=1.0)
        assert result["scalar"] == 1.0
        assert result["high_volatility"] is False

    def test_vix_above_22_reduces_sizing(self):
        result = compute_volatility_scalar(vix=32.0)
        assert result["vix_breach"] is True
        assert result["scalar"] < 1.0
        assert result["scalar"] == result["volatility_floor"]  # 32 => 0.25

    def test_vix_linear_taper(self):
        at22 = compute_volatility_scalar(vix=22.0)["scalar"]
        at27 = compute_volatility_scalar(vix=27.0)["scalar"]
        at32 = compute_volatility_scalar(vix=32.0)["scalar"]
        assert at22 == 1.0
        assert at27 == pytest.approx(0.625)  # 5 pts above threshold
        assert at32 == pytest.approx(0.25)   # at the floor

    def test_atr_2x_expansion_reduces_sizing(self):
        result = compute_volatility_scalar(atr_current=2.0, atr_average=1.0)
        assert result["atr_expansion"] is False  # strict > threshold
        assert result["scalar"] == 1.0

        result = compute_volatility_scalar(atr_current=2.6, atr_average=1.0)
        assert result["atr_expansion"] is True
        assert result["atr_ratio"] == pytest.approx(2.6)
        assert result["scalar"] == pytest.approx(1 - 0.6 * 0.75)

        result = compute_volatility_scalar(atr_current=3.0, atr_average=1.0)
        assert result["scalar"] == result["volatility_floor"]

    def test_combined_product(self):
        result = compute_volatility_scalar(vix=25.0, atr_current=2.5, atr_average=1.0)
        assert result["scalar"] == pytest.approx(0.775 * 0.625, abs=1e-3)

    def test_atr_leg_fails_open_when_missing(self):
        result = compute_volatility_scalar(vix=None, atr_current=None, atr_average=None)
        assert result["scalar"] == 1.0
        assert result["high_volatility"] is False

    def test_scaled_position_size(self, guard):
        result = guard.scaled_position_size(
            100_000, vix=32.0, atr_current=1.0, atr_average=1.0
        )
        assert result["scaled_value"] == 25_000.0


class TestMaxAllocation:
    def test_buy_blocked_beyond_15pct(self, guard):
        # existing position 10% + new order 6% = 16% > 15% cap
        result = guard.check_order(
            order_value=60_000, ticker="RELIANCE.NS", current_ticker_value=100_000
        )
        assert result["approved"] is False
        assert any(c["check"] == "max_allocation" and not c["passed"] for c in result["checks"])

    def test_buy_ok_at_15pct(self, guard):
        result = guard.check_order(
            order_value=50_000, ticker="RELIANCE.NS", current_ticker_value=100_000
        )
        assert result["approved"] is True

    def test_sell_never_blocked_by_allocation(self, guard):
        result = guard.check_order(
            order_value=400_000, ticker="RELIANCE.NS", current_ticker_value=200_000,
            is_buy=False,
        )
        assert result["approved"] is True

    def test_max_allocation_value(self, guard):
        assert guard.max_allocation_value() == 150_000.0


class TestState:
    def test_status_shape(self, guard):
        guard.update_equity(950_000)
        status = guard.get_status()
        assert status["equity"] == 950_000.0
        assert status["peak_equity"] == 1_000_000.0
        assert status["drawdown_pct"] == pytest.approx(0.05)
        assert status["daily_loss_pct"] == pytest.approx(-0.05)
        assert status["max_allocation_per_stock"] == 142_500.0
        assert status["override_token_configured"] is True

    def test_equity_recovery_updates_peak(self, guard):
        guard.update_equity(900_000)
        guard.update_equity(980_000)
        assert guard.peak_equity == 1_000_000.0
        assert guard.drawdown_pct() == pytest.approx(0.02)

    def test_state_file_written(self, guard, tmp_path):
        guard.update_equity(950_000)
        path = tmp_path / "risk_guard_state.json"
        assert path.exists()
        state = json.loads(path.read_text())
        assert state["equity"] == 950_000.0

    def test_session_date_valid(self, guard):
        assert isinstance(_ist_date(), str)
        assert guard.session_date == _ist_date()

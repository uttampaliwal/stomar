"""Tests for src/paper_trader.py."""

import pytest

from src.trading.engine import OrderSide, OrderType, OrderStatus
from src.trading.paper_trader import PaperTrader, Position
from src.trading.risk_controls import RiskLimits


def _trader(capital=100_000):
    limits = RiskLimits(max_position_pct=0.5, max_drawdown_pct=0.15)
    return PaperTrader(initial_capital=capital, slippage_bps=0, risk_limits=limits)


# ── Position ──

class TestPosition:
    def test_market_value(self):
        p = Position("T", quantity=10, avg_cost=100, current_price=110)
        assert p.market_value == 1100

    def test_pnl(self):
        p = Position("T", quantity=10, avg_cost=100, current_price=110)
        assert p.pnl == 100

    def test_pnl_pct(self):
        p = Position("T", quantity=10, avg_cost=100, current_price=110)
        assert p.pnl_pct == pytest.approx(0.1)

    def test_pnl_negative(self):
        p = Position("T", quantity=10, avg_cost=100, current_price=90)
        assert p.pnl == -100


# ── Basic Buy/Sell ──

class TestBasicTrading:
    def test_buy_order_submitted(self):
        t = _trader()
        order = t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        assert order.status == OrderStatus.PENDING

    def test_buy_fills_reduces_cash(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert t.cash < 100_000

    def test_buy_creates_position(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert "TEST.NS" in t.positions
        assert t.positions["TEST.NS"].quantity == 10

    def test_sell_fills_increases_cash(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        cash_after_buy = t.cash
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert t.cash > cash_after_buy

    def test_sell_removes_position(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert "TEST.NS" not in t.positions


# ── Risk Checks ──

class TestRiskIntegration:
    def test_large_order_rejected(self):
        t = _trader(capital=100_000)
        order = t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 100,
                              price=60_000)
        assert order.status == OrderStatus.REJECTED

    def test_halt_blocks_orders(self):
        t = _trader()
        t.risk_controller.halted = True
        order = t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        assert order.status == OrderStatus.REJECTED


# ── Trade Log ──

class TestTradeLog:
    def test_trade_recorded(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert len(t.trade_log) == 1
        assert t.trade_log[0].ticker == "TEST.NS"
        assert t.trade_log[0].side == "BUY"

    def test_cumulative_pnl(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1010, h=1020, low=1000, c=1015)
        assert t.trade_log[-1].side == "SELL"
        assert len(t.trade_log) == 2


# ── Summary ──

class TestSummary:
    def test_summary_keys(self):
        t = _trader()
        summary = t.get_summary()
        assert "initial_capital" in summary
        assert "current_equity" in summary
        assert "total_trades" in summary
        assert "open_positions" in summary

    def test_equity_starts_at_capital(self):
        t = _trader()
        assert t.get_equity() == 100_000

    def test_export_session(self, tmp_path):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        path = tmp_path / "session.json"
        t.export_session(str(path))
        assert path.exists()


# ── Reset ──

class TestReset:
    def test_reset_clears_all(self):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        t.reset()
        assert t.cash == 100_000
        assert len(t.positions) == 0
        assert len(t.trade_log) == 0
        assert t.get_equity() == 100_000


# ── Save/Load State ──

class TestStatePersistence:
    def test_save_state(self, tmp_path):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        path = tmp_path / "state.json"
        t.save_state(str(path))
        assert path.exists()

    def test_load_state(self, tmp_path):
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        path = tmp_path / "state.json"
        t.save_state(str(path))

        t2 = _trader()
        loaded = t2.load_state(str(path))
        assert loaded is True
        assert t2.cash == t.cash
        assert "TEST.NS" in t2.positions

    def test_load_nonexistent_returns_false(self, tmp_path):
        t = _trader()
        result = t.load_state(str(tmp_path / "nope.json"))
        assert result is False

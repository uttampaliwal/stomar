"""Tests for src/paper_trader.py."""

import pytest

from src.trading.engine import OrderSide, OrderType, OrderStatus
from src.trading.paper_trader import PaperTrader, Position
from src.trading.risk_controls import RiskController, RiskLimits


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


# ── Consecutive-Loss Circuit Breaker ──

class TestConsecutiveLossCircuitBreaker:
    def _trader(self, tmp_path, capital=100_000):
        t = _trader(capital=capital)
        t.risk_controller = RiskController(
            initial_capital=capital,
            kill_switch_file=tmp_path / "kill_switch.json",  # avoid real data dir
        )
        return t

    def test_entry_fill_does_not_reset_streak(self, tmp_path):
        t = self._trader(tmp_path)
        t.risk_controller.consecutive_losses = 4
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        assert t.risk_controller.consecutive_losses == 4

    def test_losing_close_trips_kill_switch(self, tmp_path):
        t = self._trader(tmp_path)
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        t.risk_controller.consecutive_losses = 4
        t.execute_market_trade("TEST.NS", OrderSide.SELL, 5, 95.0)
        assert t.risk_controller.consecutive_losses == 5
        assert t.risk_controller.halted is True
        result = t.risk_controller.check_order(
            500, 500, ticker="TEST.NS", holdings={}, prices={}
        )
        assert result["approved"] is False

    def test_winning_close_resets_streak(self, tmp_path):
        t = self._trader(tmp_path)
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        t.risk_controller.consecutive_losses = 4
        t.execute_market_trade("TEST.NS", OrderSide.SELL, 5, 110.0)
        assert t.risk_controller.consecutive_losses == 0


# ── Stop-Loss Execution ──

class TestStopExecution:
    def test_stop_market_sell_fills_when_price_crosses(self):
        t = _trader()
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET, 10,
                      stop_price=95.0)
        assert len(t.engine.get_pending("TEST.NS")) == 1

        records = t.check_stops("TEST.NS", 94.0)
        assert len(records) == 1
        assert records[0].side == "SELL"
        assert records[0].quantity == 10
        assert "TEST.NS" not in t.positions
        assert not t.engine.get_pending("TEST.NS")

    def test_stop_not_filled_above_level(self):
        t = _trader()
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET, 10,
                      stop_price=95.0)
        assert t.check_stops("TEST.NS", 96.0) == []
        assert len(t.engine.get_pending("TEST.NS")) == 1

    def test_stop_uses_position_price_when_omitted(self):
        t = _trader()
        t.execute_market_trade("TEST.NS", OrderSide.BUY, 10, 100.0)
        t.place_order("TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET, 10,
                      stop_price=95.0)
        t.positions["TEST.NS"].current_price = 93.0
        records = t.check_stops("TEST.NS")
        assert len(records) == 1

    def test_check_stops_all_tickers_only_triggers_crossed(self):
        t = _trader()
        t.execute_market_trade("A.NS", OrderSide.BUY, 10, 100.0)
        t.execute_market_trade("B.NS", OrderSide.BUY, 10, 200.0)
        t.place_order("A.NS", OrderSide.SELL, OrderType.STOP_MARKET, 10,
                      stop_price=95.0)
        t.place_order("B.NS", OrderSide.SELL, OrderType.STOP_MARKET, 10,
                      stop_price=195.0)
        t.positions["A.NS"].current_price = 90.0
        t.positions["B.NS"].current_price = 210.0

        records = t.check_stops()
        assert len(records) == 1
        assert "A.NS" not in t.positions
        assert "B.NS" in t.positions

    def test_no_pending_stops_returns_empty(self):
        t = _trader()
        assert t.check_stops("TEST.NS", 100.0) == []


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

    def test_state_restores_capital_streak_and_halt(self, tmp_path):
        """O1: restart must not reopen the loss budget (capital+streak+halt)."""
        t = _trader()
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        t.risk_controller.update_consecutive_losses(True)
        t.risk_controller.update_consecutive_losses(True)
        t.risk_controller.halted = True
        path = tmp_path / "state.json"
        t.save_state(str(path))

        t2 = PaperTrader(initial_capital=999_999)
        assert t2.load_state(str(path)) is True
        assert t2.initial_capital == t.initial_capital
        assert t2.risk_controller.consecutive_losses == 2
        assert t2.risk_controller.halted is True

    def test_on_bar_fill_auto_saves_state(self, tmp_path, monkeypatch):
        """O1: stop/limit fills via on_bar() must persist (not just market trades)."""
        t = _trader()
        t.save_state(str(tmp_path / "seed.json"))
        saved = []
        monkeypatch.setattr(
            t, "save_state", lambda path=None: saved.append(path or "auto")
        )
        t.place_order("TEST.NS", OrderSide.BUY, OrderType.MARKET, 10)
        t.on_bar("TEST.NS", o=1000, h=1010, low=990, c=1005)
        assert saved, "on_bar fills must trigger a state save"

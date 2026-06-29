"""Tests for src/engine.py."""

import pytest

from src.engine import (
    ExecutionEngine, Order, OrderSide, OrderType, OrderStatus, Bar,
    FixedSlippage, VolumeSlippage, AdaptiveSlippage,
)


def _bar(ticker="TEST.NS", ts="2025-01-01", o=100, h=105, l=95, c=102, v=1_000_000):
    return Bar(ticker, ts, o, h, l, c, v)


# ── Order ──

class TestOrder:
    def test_defaults(self):
        o = Order(order_id="", ticker="T", side=OrderSide.BUY,
                  order_type=OrderType.MARKET, quantity=10)
        assert o.status == OrderStatus.PENDING
        assert o.filled_price == 0.0

    def test_sell_order(self):
        o = Order(order_id="", ticker="T", side=OrderSide.SELL,
                  order_type=OrderType.MARKET, quantity=10)
        assert o.side == OrderSide.SELL


# ── ExecutionEngine ──

class TestSubmitOrder:
    def test_gets_id(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        order = Order(order_id="", ticker="T", side=OrderSide.BUY,
                      order_type=OrderType.MARKET, quantity=10)
        result = engine.submit_order(order)
        assert result.order_id == "ORD-000001"
        assert result.status == OrderStatus.PENDING

    def test_increments_id(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "T", OrderSide.BUY, OrderType.MARKET, 10))
        engine.submit_order(Order("", "T", OrderSide.BUY, OrderType.MARKET, 10))
        assert engine.order_counter == 2

    def test_pending_queue(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "T", OrderSide.BUY, OrderType.MARKET, 10))
        assert len(engine.pending_orders) == 1


class TestMarketOrderFills:
    def test_fills_at_open(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0
        assert filled[0].filled_quantity == 10

    def test_sell_fills_at_open(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0

    def test_other_ticker_stays_pending(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "OTHER.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(ticker="TEST.NS"))
        assert len(filled) == 0
        assert len(engine.pending_orders) == 1


class TestLimitOrderFills:
    def test_limit_buy_fills_when_low_touches(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=98))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price <= 98.0  # Fill at limit or better

    def test_limit_buy_not_filled_when_price_high(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=90))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 0

    def test_limit_sell_fills_when_high_touches(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.LIMIT, 10, price=103))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price >= 103.0

    def test_limit_sell_not_filled_when_price_low(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.LIMIT, 10, price=110))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 0


class TestStopOrderFills:
    def test_stop_loss_sell_triggers(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_LOSS,
                                  10, stop_price=97))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 97.0

    def test_stop_loss_not_triggered_when_price_high(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_LOSS,
                                  10, stop_price=90))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 0

    def test_stop_market_sell_fills_at_open(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET,
                                  10, stop_price=97))
        filled = engine.on_bar(_bar(o=100, h=105, l=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0  # Market fill at open


class TestCancelAll:
    def test_cancel_all(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "A", OrderSide.BUY, OrderType.MARKET, 10))
        engine.submit_order(Order("", "B", OrderSide.BUY, OrderType.MARKET, 10))
        cancelled = engine.cancel_all()
        assert len(cancelled) == 2
        assert len(engine.pending_orders) == 0

    def test_cancel_by_ticker(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "A", OrderSide.BUY, OrderType.MARKET, 10))
        engine.submit_order(Order("", "B", OrderSide.BUY, OrderType.MARKET, 10))
        cancelled = engine.cancel_all(ticker="A")
        assert len(cancelled) == 1
        assert len(engine.pending_orders) == 1


class TestSlippage:
    def test_slippage_applied_to_buy(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0.005))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100))
        assert filled[0].filled_price > 100.0

    def test_slippage_applied_to_sell(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0.005))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100))
        assert filled[0].filled_price < 100.0

    def test_zero_slippage(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100))
        assert filled[0].filled_price == 100.0


# ── Slippage Models ──

class TestFixedSlippage:
    def test_returns_price_times_rate(self):
        s = FixedSlippage(0.001)
        assert s.calculate(OrderSide.BUY, 100, 1_000_000) == 0.1


class TestVolumeSlippage:
    def test_high_volume_low_slippage(self):
        s = VolumeSlippage(0.002, 1_000_000)
        result = s.calculate(OrderSide.BUY, 100, 2_000_000)
        assert result < 100 * 0.002

    def test_low_volume_high_slippage(self):
        s = VolumeSlippage(0.002, 1_000_000)
        result = s.calculate(OrderSide.BUY, 100, 100_000)
        assert result > 100 * 0.002

    def test_capped_at_3x(self):
        s = VolumeSlippage(0.002, 1_000_000)
        result = s.calculate(OrderSide.BUY, 100, 1)
        expected = 100 * 0.002 * 3.0
        assert result == pytest.approx(expected, rel=1e-9)


class TestAdaptiveSlippage:
    def test_increases_with_order_size(self):
        s = AdaptiveSlippage(0.001, 0.1)
        small = s.calculate(OrderSide.BUY, 100, 1_000_000, order_value=10_000)
        large = s.calculate(OrderSide.BUY, 100, 1_000_000, order_value=100_000)
        assert large > small

    def test_base_slippage_always_present(self):
        s = AdaptiveSlippage(0.001, 0.1)
        result = s.calculate(OrderSide.BUY, 100, 1_000_000, order_value=0)
        assert result > 0

"""Tests for src/engine.py."""

import numpy as np
import pytest
from unittest.mock import patch

from src.trading.engine import (
    ExecutionEngine, Order, OrderSide, OrderType, OrderStatus, Bar,
    FixedSlippage, VolumeSlippage, AdaptiveSlippage,
)


def _bar(ticker="TEST.NS", ts="2025-01-01", o=100, h=105, lo=95, c=102, v=1_000_000):
    return Bar(ticker, ts, o, h, lo, c, v)


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
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0
        assert filled[0].filled_quantity == 10

    def test_sell_fills_at_open(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
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
        engine = ExecutionEngine(slippage_model=FixedSlippage(0), fill_probability=1.0)
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=101))
        with patch.object(engine, "rng") as mock_rng:
            mock_rng.random.return_value = 0.0
            filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price <= 101.0

    def test_limit_buy_not_filled_when_price_high(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0), fill_probability=1.0)
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=85))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 0

    def test_limit_sell_fills_when_high_touches(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0), fill_probability=1.0)
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.LIMIT, 10, price=103))
        with patch.object(engine, "rng") as mock_rng:
            mock_rng.random.return_value = 0.0
            filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price >= 103.0

    def test_limit_sell_not_filled_when_price_low(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0), fill_probability=1.0)
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.LIMIT, 10, price=120))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 0


class TestStopOrderFills:
    def test_stop_loss_sell_triggers(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_LOSS,
                                  10, stop_price=97))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 97.0

    def test_stop_loss_not_triggered_when_price_high(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_LOSS,
                                  10, stop_price=90))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 0

    def test_stop_market_sell_fills_at_open(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET,
                                  10, stop_price=97))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0  # Market fill at open


class TestDeterminism:
    def test_default_seed_reproduces_limit_fill_sequence(self):
        # A limit order near the close has prob < 1.0, so the RNG decides
        # whether it fills. Fresh engines must replay identical outcomes.
        def run():
            engine = ExecutionEngine(slippage_model=FixedSlippage(0))
            outcomes = []
            for _ in range(50):
                engine.submit_order(
                    Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10,
                          price=98.5)
                )
                outcomes.append(len(engine.on_bar(_bar(o=100, h=105, lo=95, c=102))))
            return outcomes

        assert run() == run()

    def test_injected_rng_is_used(self):
        import numpy as np
        rng = np.random.default_rng(7)
        engine = ExecutionEngine(slippage_model=FixedSlippage(0), rng=rng)
        assert engine.rng is rng


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


# ── Fill Probability ──

class TestFillProbability:
    def test_market_always_fills(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1

    def test_limit_near_close_high_probability(self):
        # Deterministic: one fresh RNG stream per trial, so the ~70% fill
        # rate is sampled across seeds instead of the (now-seeded) default.
        results = []
        for i in range(20):
            engine = ExecutionEngine(slippage_model=FixedSlippage(0),
                                     rng=np.random.default_rng(i))
            engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=101))
            filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
            results.append(len(filled) > 0)
        assert sum(results) >= 10  # ~70% probability, should usually pass

    def test_limit_far_low_probability(self):
        results = []
        for i in range(20):
            engine = ExecutionEngine(slippage_model=FixedSlippage(0),
                                     rng=np.random.default_rng(i))
            engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, 10, price=85))
            filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=102))
            results.append(len(filled) > 0)
        assert sum(results) < 15


# ── State restore ──

class TestRestoreState:
    def test_restore_roundtrip(self):
        engine = ExecutionEngine(slippage_model=FixedSlippage(0))
        engine.submit_order(Order("", "T", OrderSide.BUY, OrderType.MARKET, 10))
        state = engine.get_state()
        restored = ExecutionEngine(slippage_model=FixedSlippage(0))
        restored.restore_state(state)
        assert restored.order_counter == engine.order_counter
        assert [o.side for o in restored.pending_orders] == [OrderSide.BUY]
        assert [o.order_type for o in restored.pending_orders] == [OrderType.MARKET]

    def test_invalid_side_defaults_to_buy(self):
        engine = ExecutionEngine()
        d = {"order_id": "x", "ticker": "T", "side": "SIDEWAYS",
             "order_type": "MARKET", "quantity": 5}
        order = engine._order_from_dict(d)
        assert order.side == OrderSide.BUY
        assert order.quantity == 5

    def test_invalid_order_type_defaults_to_market(self):
        engine = ExecutionEngine()
        d = {"order_id": "x", "ticker": "T", "side": "SELL",
             "order_type": "FROBNICATE", "quantity": 5}
        order = engine._order_from_dict(d)
        assert order.order_type == OrderType.MARKET

    def test_invalid_status_defaults_to_pending(self):
        engine = ExecutionEngine()
        d = {"order_id": "x", "ticker": "T", "side": "BUY",
             "order_type": "LIMIT", "quantity": 5, "status": "DONE"}
        order = engine._order_from_dict(d)
        assert order.status == OrderStatus.PENDING

    def test_corrupted_state_restores_without_raising(self):
        engine = ExecutionEngine()
        state = {"order_counter": 3, "pending_orders": [
            {"order_id": "a", "ticker": "T", "side": "HOLD",
             "order_type": "MARKET", "quantity": 5},
        ], "filled_orders": [], "rejected_orders": []}
        engine.restore_state(state)
        assert len(engine.pending_orders) == 1
        assert engine.pending_orders[0].side == OrderSide.BUY


# ── BacktestExecutionSimulator ──

class TestBacktestExecutionSimulator:
    def test_simulate_fill_buy_adds_slippage(self):
        from src.trading.engine import BacktestExecutionSimulator
        sim = BacktestExecutionSimulator(slippage_model="fixed")
        fill = sim.simulate_fill(OrderSide.BUY, 100, 1_000_000)
        assert fill > 100

    def test_simulate_fill_sell_subtracts_slippage(self):
        from src.trading.engine import BacktestExecutionSimulator
        sim = BacktestExecutionSimulator(slippage_model="fixed")
        fill = sim.simulate_fill(OrderSide.SELL, 100, 1_000_000)
        assert fill < 100

    def test_fill_prob_market_always_one(self):
        from src.trading.engine import BacktestExecutionSimulator
        sim = BacktestExecutionSimulator()
        assert sim.estimate_fill_probability(OrderType.MARKET, 100, 100) == 1.0

    def test_fill_prob_limit_near_high(self):
        from src.trading.engine import BacktestExecutionSimulator
        sim = BacktestExecutionSimulator()
        prob = sim.estimate_fill_probability(OrderType.LIMIT, 101, 102)
        assert prob >= 0.7

    def test_fill_prob_limit_far_low(self):
        from src.trading.engine import BacktestExecutionSimulator
        sim = BacktestExecutionSimulator()
        prob = sim.estimate_fill_probability(OrderType.LIMIT, 85, 102)
        assert prob <= 0.1

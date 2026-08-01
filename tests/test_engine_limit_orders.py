"""Edge-case tests for the execution engine's limit/stop order logic.

The basic suite (test_engine.py) covers market fills and simple limit/stop
triggers. This file pins down the fill-price *semantics* that matter for
backtest-live parity:

* LIMIT fills at the limit price when the bar trades through it, but at
  the (gap) open when the bar opens through the limit — never better than
  the limit, never worse than the open.
* STOP_MARKET triggers a market fill at the bar OPEN, while STOP_LOSS
  fills at the stop price (limit-style). Both sides (buy/sell) are covered.
* Partial fills are NOT implemented: an order either fills in full or
  stays PENDING — PARTIALLY_FILLED must never appear.
* Invalid bars (high < low, zero/negative prices) are rejected without
  touching pending orders.
"""

from unittest.mock import patch

from src.trading.engine import (
    ExecutionEngine, Order, OrderSide, OrderType, OrderStatus, Bar,
    FixedSlippage,
)


def _bar(ticker="TEST.NS", ts="2025-01-01", o=100.0, h=105.0, lo=95.0, c=102.0, v=1_000_000):
    return Bar(ticker, ts, o, h, lo, c, v)


def _make_engine():
    return ExecutionEngine(slippage_model=FixedSlippage(0), fill_probability=1.0)


def _buy_limit(price, qty=10):
    return Order("", "TEST.NS", OrderSide.BUY, OrderType.LIMIT, qty, price=price)


def _sell_limit(price, qty=10):
    return Order("", "TEST.NS", OrderSide.SELL, OrderType.LIMIT, qty, price=price)


def _buy_stop(stop, qty=10):
    return Order("", "TEST.NS", OrderSide.BUY, OrderType.STOP_MARKET, qty, stop_price=stop)


def _sell_stop(stop, qty=10):
    return Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_MARKET, qty, stop_price=stop)


def _run(engine, bar):
    with patch("src.trading.engine.random.random", return_value=0.0):
        return engine.on_bar(bar)


# ── LIMIT BUY: fill price = min(limit, open) ───────────────────────────────

class TestLimitBuyFillPrice:
    def test_fills_at_limit_when_bar_touches_below_open(self):
        # open 100 > limit 98, low 95 <= 98 -> fill at the limit, not the open
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 98.0
        assert filled[0].status == OrderStatus.FILLED

    def test_gap_down_fills_at_open_below_limit(self):
        # open 90 < limit 98 -> the bar opens through the limit; fill at open
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = _run(engine, _bar(o=90, h=96, lo=88, c=92))
        assert len(filled) == 1
        assert filled[0].filled_price == 90.0

    def test_never_fills_above_limit_price(self):
        for open_, low in [(100, 95), (90, 88), (100, 97)]:
            engine = _make_engine()
            engine.submit_order(_buy_limit(98))
            filled = _run(engine, _bar(o=open_, h=105, lo=low, c=102))
            assert filled[0].filled_price <= 98.0

    def test_exact_touch_at_limit_price(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(100))
        filled = _run(engine, _bar(o=101, h=105, lo=100, c=102))
        assert filled[0].filled_price == 100.0

    def test_not_filled_when_low_above_limit(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = _run(engine, _bar(o=100, h=105, lo=101, c=102))
        assert filled == []
        assert engine.get_pending()[0].status == OrderStatus.PENDING


# ── LIMIT SELL: fill price = max(limit, open) ──────────────────────────────

class TestLimitSellFillPrice:
    def test_fills_at_limit_when_bar_touches_above_open(self):
        engine = _make_engine()
        engine.submit_order(_sell_limit(103))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 103.0

    def test_gap_up_fills_at_open_above_limit(self):
        engine = _make_engine()
        engine.submit_order(_sell_limit(103))
        filled = _run(engine, _bar(o=110, h=112, lo=108, c=111))
        assert len(filled) == 1
        assert filled[0].filled_price == 110.0

    def test_never_fills_below_limit_price(self):
        for open_, high in [(100, 105), (110, 112), (100, 103)]:
            engine = _make_engine()
            engine.submit_order(_sell_limit(103))
            filled = _run(engine, _bar(o=open_, h=high, lo=95, c=102))
            assert filled[0].filled_price >= 103.0

    def test_exact_touch_at_limit_price(self):
        engine = _make_engine()
        engine.submit_order(_sell_limit(100))
        filled = _run(engine, _bar(o=99, h=100, lo=95, c=102))
        assert filled[0].filled_price == 100.0

    def test_not_filled_when_high_below_limit(self):
        engine = _make_engine()
        engine.submit_order(_sell_limit(103))
        filled = _run(engine, _bar(o=100, h=102, lo=95, c=102))
        assert filled == []
        assert engine.get_pending()[0].status == OrderStatus.PENDING


# ── STOP_MARKET: market-on-trigger, fill at the bar OPEN ───────────────────

class TestStopMarket:
    def test_sell_triggers_at_open_when_low_touches_stop(self):
        engine = _make_engine()
        engine.submit_order(_sell_stop(97))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0

    def test_sell_gap_down_fills_at_open_below_stop(self):
        # open gaps below the stop -> market order executes at the open
        engine = _make_engine()
        engine.submit_order(_sell_stop(97))
        filled = _run(engine, _bar(o=92, h=96, lo=90, c=93))
        assert len(filled) == 1
        assert filled[0].filled_price == 92.0

    def test_buy_triggers_at_open_when_high_touches_stop(self):
        engine = _make_engine()
        engine.submit_order(_buy_stop(103))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 100.0

    def test_buy_gap_up_fills_at_open_above_stop(self):
        engine = _make_engine()
        engine.submit_order(_buy_stop(103))
        filled = _run(engine, _bar(o=108, h=110, lo=106, c=109))
        assert len(filled) == 1
        assert filled[0].filled_price == 108.0

    def test_not_triggered_when_bar_never_touches_stop(self):
        engine = _make_engine()
        engine.submit_order(_buy_stop(103))
        filled = _run(engine, _bar(o=100, h=102, lo=95, c=102))
        assert filled == []
        assert engine.get_pending()[0].status == OrderStatus.PENDING

    def test_stop_market_fills_at_open_not_stop_price(self):
        # same trigger condition as STOP_LOSS, but the fill price differs:
        # STOP_LOSS is limit-style (stop price), STOP_MARKET is market (open)
        engine = _make_engine()
        engine.submit_order(Order("", "TEST.NS", OrderSide.SELL, OrderType.STOP_LOSS,
                                  10, stop_price=97))
        engine.submit_order(_sell_stop(97))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        fills = {o.order_type: o.filled_price for o in filled}
        assert fills[OrderType.STOP_LOSS] == 97.0
        assert fills[OrderType.STOP_MARKET] == 100.0


# ── STOP_LOSS (limit-style) — buy side untested in the basic suite ─────────

class TestStopLossBuy:
    def test_buy_triggers_at_stop_price(self):
        engine = _make_engine()
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.STOP_LOSS,
                                  10, stop_price=103))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_price == 103.0

    def test_buy_not_triggered_when_high_below_stop(self):
        engine = _make_engine()
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.STOP_LOSS,
                                  10, stop_price=103))
        filled = _run(engine, _bar(o=100, h=102, lo=95, c=102))
        assert filled == []
        assert engine.get_pending()[0].status == OrderStatus.PENDING


# ── Partial fills are not implemented ──────────────────────────────────────

class TestPartialFillNotImplemented:
    def test_order_fills_in_full_or_stays_pending(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98, qty=25))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 1
        assert filled[0].filled_quantity == 25
        assert filled[0].status == OrderStatus.FILLED

    def test_partial_status_never_assigned(self):
        # a spread of limit prices against one bar: filled or pending, never partial
        engine = _make_engine()
        for price in (99.0, 98.0, 97.0, 90.0, 80.0):
            engine.submit_order(_buy_limit(price, qty=10))
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(filled) == 3
        for order in engine.get_filled() + engine.get_pending():
            assert order.status != OrderStatus.PARTIALLY_FILLED
            assert order.filled_quantity in (0, 10)

    def test_unfilled_order_survives_across_bars(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        _run(engine, _bar(o=100, h=105, lo=99, c=102))
        assert engine.get_pending()[0].status == OrderStatus.PENDING
        filled = _run(engine, _bar(ts="2025-01-02", o=99, h=101, lo=97, c=100))
        assert filled[0].filled_price == 98.0


# ── Invalid bars are rejected without touching pending orders ──────────────

class TestInvalidBarRejection:
    def test_high_below_low_rejected(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = engine.on_bar(_bar(o=100, h=95, lo=105, c=102))
        assert filled == []
        assert len(engine.get_pending()) == 1

    def test_zero_close_rejected(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = engine.on_bar(_bar(o=100, h=105, lo=95, c=0))
        assert filled == []
        assert len(engine.get_pending()) == 1

    def test_zero_open_rejected(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = engine.on_bar(_bar(o=0, h=105, lo=95, c=102))
        assert filled == []
        assert len(engine.get_pending()) == 1

    def test_negative_prices_rejected(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        filled = engine.on_bar(_bar(o=100, h=105, lo=-95, c=-102))
        assert filled == []
        assert len(engine.get_pending()) == 1

    def test_market_order_not_filled_on_invalid_bar(self):
        engine = _make_engine()
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        engine.on_bar(_bar(o=100, h=95, lo=105, c=102))
        assert len(engine.get_pending()) == 1

    def test_invalid_bar_does_not_cancel_orders(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        engine.on_bar(_bar(o=100, h=95, lo=105, c=102))
        filled = _run(engine, _bar(ts="2025-01-02", o=99, h=101, lo=97, c=100))
        assert len(filled) == 1
        assert filled[0].filled_price == 98.0

    def test_doji_bar_high_equals_low_is_valid(self):
        engine = _make_engine()
        engine.submit_order(Order("", "TEST.NS", OrderSide.BUY, OrderType.MARKET, 10))
        filled = _run(engine, _bar(o=100, h=100, lo=100, c=100))
        assert len(filled) == 1


# ── Lifecycle: filled orders are not reprocessed ───────────────────────────

class TestOrderLifecycle:
    def test_filled_order_not_reprocessed_on_next_bar(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert len(engine.get_filled()) == 1
        filled = _run(engine, _bar(ts="2025-01-02", o=90, h=92, lo=88, c=91))
        assert filled == []
        assert len(engine.get_filled()) == 1

    def test_orders_for_other_tickers_skip_bar(self):
        engine = _make_engine()
        engine.submit_order(_buy_limit(98))
        other = Order("", "OTHER.NS", OrderSide.BUY, OrderType.MARKET, 10)
        engine.submit_order(other)
        filled = _run(engine, _bar(o=100, h=105, lo=95, c=102))
        assert [o.order_id for o in filled] == ["ORD-000001"]
        assert engine.get_pending(ticker="OTHER.NS") == [other]

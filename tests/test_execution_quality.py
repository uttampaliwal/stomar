"""Tests for src/execution_quality.py."""

import pytest

from src.execution_quality import ExecutionQualityAnalyzer, _safe_mean


def _analyzer():
    return ExecutionQualityAnalyzer()


# ── Basic ──

class TestEmptyAnalyzer:
    def test_empty_report(self):
        eq = _analyzer()
        report = eq.analyze()
        assert report["total_fills"] == 0
        assert report["fill_rate"] == 1.0

    def test_per_ticker_empty(self):
        eq = _analyzer()
        assert eq.per_ticker() == {}


class TestSafeMean:
    def test_empty(self):
        assert _safe_mean([]) == 0.0

    def test_normal(self):
        assert _safe_mean([1.0, 2.0, 3.0]) == 2.0


# ── Fills ──

class TestFills:
    def test_single_buy(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2505)
        assert len(eq.fills) == 1

    def test_buy_slippage_positive(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2505)
        assert eq.fills[0].slippage == 5.0

    def test_sell_slippage_positive(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "SELL", 2500, 10, 2495)
        assert eq.fills[0].slippage == 5.0

    def test_fill_cost(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        assert eq.fills[0].fill_cost == 25000


# ── Analysis ──

class TestAnalysis:
    def test_buy_slippage_avg(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2510)
        report = eq.analyze()
        assert report["avg_slippage"] == pytest.approx(5.0)

    def test_sell_slippage_avg(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "SELL", 2500, 10, 2500)
        eq.add_fill("RELIANCE.NS", "SELL", 2500, 10, 2490)
        report = eq.analyze()
        assert report["avg_slippage"] == pytest.approx(5.0)

    def test_vwap_slippage(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2505)
        eq.set_vwap("RELIANCE.NS", 2503)
        report = eq.analyze()
        assert report["vwap_slippage"] == pytest.approx(2.0)

    def test_open_slippage(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2505)
        eq.set_open("RELIANCE.NS", 2490)
        report = eq.analyze()
        assert report["open_slippage"] == pytest.approx(15.0)

    def test_total_volume(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        eq.add_fill("RELIANCE.NS", "SELL", 2500, 5, 2500)
        report = eq.analyze()
        assert report["total_volume"] == 15

    def test_buy_sell_counts(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        eq.add_fill("RELIANCE.NS", "SELL", 2500, 10, 2500)
        report = eq.analyze()
        assert report["buy_count"] == 2
        assert report["sell_count"] == 1


# ── Per-Ticker ──

class TestPerTicker:
    def test_multiple_tickers(self):
        eq = _analyzer()
        eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2500)
        eq.add_fill("TCS.NS", "BUY", 3500, 5, 3500)
        per = eq.per_ticker()
        assert "RELIANCE.NS" in per
        assert "TCS.NS" in per
        assert per["RELIANCE.NS"]["fills"] == 1
        assert per["TCS.NS"]["volume"] == 5

"""Tests for src/mf_tracker.py — MF portfolio tracker."""

import os

import numpy as np
import pandas as pd
import pytest

from src.signals.mf_tracker import MFTracker


@pytest.fixture
def tracker():
    return MFTracker()


@pytest.fixture
def tracker_with_holdings(tracker):
    """Tracker with 2 holdings and mock NAV cache."""
    tracker.add_holding("HDFC", units=100, avg_nav=100.0,
                        fund_name="HDFC Flex Cap", category="Equity")
    tracker.add_holding("ICICI", units=50, avg_nav=200.0,
                        fund_name="ICICI Prudential", category="Debt")
    # Mock NAV data
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=252, freq="B")
    nav_hdfc = pd.Series(100 + np.cumsum(np.random.randn(252) * 0.5), index=dates)
    nav_icici = pd.Series(200 + np.cumsum(np.random.randn(252) * 1.0), index=dates)
    tracker.nav_cache["HDFC"] = nav_hdfc
    tracker.nav_cache["ICICI"] = nav_icici
    return tracker


class TestAddHolding:
    def test_add_single(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0, fund_name="HDFC Flex Cap")
        assert "HDFC" in tracker.holdings
        assert tracker.holdings["HDFC"].units == 100

    def test_add_multiple(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        tracker.add_holding("ICICI", units=50, avg_nav=200.0)
        assert len(tracker.holdings) == 2

    def test_update_existing(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        tracker.add_holding("HDFC", units=200, avg_nav=120.0)
        assert tracker.holdings["HDFC"].units == 200
        assert tracker.holdings["HDFC"].avg_nav == 120.0


class TestRemoveHolding:
    def test_removes_holding_and_nav(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        tracker.nav_cache["HDFC"] = pd.Series([100, 110])
        tracker.remove_holding("HDFC")
        assert "HDFC" not in tracker.holdings
        assert "HDFC" not in tracker.nav_cache

    def test_remove_nonexistent(self, tracker):
        tracker.remove_holding("NOPE")  # should not raise


class TestGetCurrentValue:
    def test_with_nav_cache(self, tracker_with_holdings):
        val = tracker_with_holdings.get_current_value("HDFC")
        last_nav = tracker_with_holdings.nav_cache["HDFC"].iloc[-1]
        assert val == pytest.approx(last_nav * 100)

    def test_fallback_to_avg_nav(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        val = tracker.get_current_value("HDFC")
        assert val == pytest.approx(100 * 100)

    def test_unknown_ticker(self, tracker):
        assert tracker.get_current_value("NOPE") == 0.0


class TestPortfolioMetrics:
    def test_portfolio_value(self, tracker_with_holdings):
        val = tracker_with_holdings.get_portfolio_value()
        assert val > 0

    def test_invested_value(self, tracker_with_holdings):
        invested = tracker_with_holdings.get_invested_value()
        assert invested == pytest.approx(100 * 100 + 50 * 200)

    def test_total_pnl(self, tracker_with_holdings):
        pnl = tracker_with_holdings.get_total_pnl()
        expected = tracker_with_holdings.get_portfolio_value() - tracker_with_holdings.get_invested_value()
        assert pnl == pytest.approx(expected)

    def test_total_return_pct(self, tracker_with_holdings):
        ret = tracker_with_holdings.get_total_return_pct()
        assert isinstance(ret, float)

    def test_zero_invested(self, tracker):
        assert tracker.get_total_return_pct() == 0.0


class TestXIRR:
    def test_single_xirr(self, tracker_with_holdings):
        xirr = tracker_with_holdings.compute_xirr(ticker="HDFC")
        assert isinstance(xirr, float)
        assert np.isfinite(xirr)

    def test_portfolio_xirr(self, tracker_with_holdings):
        xirr = tracker_with_holdings.compute_xirr()
        assert isinstance(xirr, float)
        assert np.isfinite(xirr)

    def test_unknown_ticker(self, tracker):
        assert tracker.compute_xirr("NOPE") == 0.0

    def test_no_nav_data(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        assert tracker.compute_xirr("HDFC") == 0.0


class TestFactorExposures:
    def test_returns_all_factors(self, tracker_with_holdings):
        factors = tracker_with_holdings.compute_factor_exposures("HDFC")
        assert "value" in factors
        assert "momentum" in factors
        assert "quality" in factors
        assert "volatility" in factors

    def test_values_in_range(self, tracker_with_holdings):
        factors = tracker_with_holdings.compute_factor_exposures("HDFC")
        assert -1.0 <= factors["value"] <= 1.0
        assert -1.0 <= factors["momentum"] <= 1.0
        assert -1.0 <= factors["quality"] <= 1.0

    def test_insufficient_data(self, tracker):
        tracker.add_holding("HDFC", units=100, avg_nav=100.0)
        factors = tracker.compute_factor_exposures("HDFC")
        assert factors == {"value": 0.0, "momentum": 0.0, "quality": 0.0}


class TestTopHoldings:
    def test_returns_sorted(self, tracker_with_holdings):
        top = tracker_with_holdings.get_top_holdings(n=2)
        assert len(top) == 2
        assert top[0]["current_value"] >= top[1]["current_value"]

    def test_keys_present(self, tracker_with_holdings):
        top = tracker_with_holdings.get_top_holdings(n=1)
        assert "ticker" in top[0]
        assert "pnl" in top[0]
        assert "return_pct" in top[0]


class TestAllocationBreakdown:
    def test_returns_weights(self, tracker_with_holdings):
        alloc = tracker_with_holdings.get_allocation_breakdown()
        assert len(alloc) > 0
        assert abs(sum(alloc.values()) - 1.0) < 0.001

    def test_empty_portfolio(self, tracker):
        assert tracker.get_allocation_breakdown() == {}


class TestConcentrationRisk:
    def test_detects_heavy_weight(self, tracker):
        tracker.add_holding("HDFC", units=1000, avg_nav=100.0, fund_name="HDFC Flex")
        tracker.add_holding("ICICI", units=10, avg_nav=100.0, fund_name="ICICI Pru")
        tracker.nav_cache["HDFC"] = pd.Series([100] * 10)
        tracker.nav_cache["ICICI"] = pd.Series([100] * 10)
        flagged = tracker.detect_concentration_risk(threshold=0.5)
        assert len(flagged) == 1
        assert flagged[0]["ticker"] == "HDFC"

    def test_no_flag_when_balanced(self, tracker):
        tracker.add_holding("A", units=50, avg_nav=100.0)
        tracker.add_holding("B", units=50, avg_nav=100.0)
        tracker.nav_cache["A"] = pd.Series([100] * 10)
        tracker.nav_cache["B"] = pd.Series([100] * 10)
        flagged = tracker.detect_concentration_risk(threshold=0.5)
        assert len(flagged) == 0

    def test_empty_portfolio(self, tracker):
        assert tracker.detect_concentration_risk() == []


class TestSaveLoadState:
    def test_roundtrip(self, tracker_with_holdings, tmp_path):
        path = str(tmp_path / "state.json")
        tracker_with_holdings.save_state(path)
        assert os.path.exists(path)

        new_tracker = MFTracker()
        assert new_tracker.load_state(path)
        assert len(new_tracker.holdings) == 2
        assert new_tracker.holdings["HDFC"].units == 100

    def test_load_nonexistent(self, tracker):
        assert tracker.load_state("nonexistent.json") is False

    def test_nav_cache_persisted(self, tracker_with_holdings, tmp_path):
        path = str(tmp_path / "state.json")
        tracker_with_holdings.save_state(path)
        new_tracker = MFTracker()
        new_tracker.load_state(path)
        assert "HDFC" in new_tracker.nav_cache
        assert len(new_tracker.nav_cache["HDFC"]) == 252

"""Tests for src/holdings.py — holdings parsing and portfolio stats."""

import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.trading.holdings import (
    parse_holdings_csv,
    compute_xirr,
    compute_portfolio_stats,
    save_holdings,
    load_holdings,
    INDIAN_MF_MAP,
)


@pytest.fixture
def sample_holdings_df():
    return pd.DataFrame({
        "name": ["HDFC Flexi Cap Fund", "ICICI Prudential Large & Mid Cap Fund", "UTI Nifty 50 Index Fund"],
        "units": [100.0, 50.0, 200.0],
        "avg_nav": [25.0, 50.0, 15.0],
        "current_nav": [27.0, 48.0, 16.0],
        "invested": [2500.0, 2500.0, 3000.0],
        "current_value": [2700.0, 2400.0, 3200.0],
        "pnl": [200.0, -100.0, 200.0],
        "net_change_pct": [2.0, -1.5, 1.0],
        "day_change_pct": [0.5, -0.3, 0.2],
    })


class TestParseHoldingsCsv:
    def test_parse_valid_csv(self, tmp_path):
        csv_content = """Instrument,Qty.,Avg. cost,LTP,Invested,Cur. val,P&L,Net chg.,Day chg.
"HDFC Flexi Cap Fund",100.00,25.00,27.00,"2,500.00","2,700.00","200.00",2.00%,0.50%
"ICICI Pru Large & Mid Cap Fund",50.00,50.00,48.00,"2,500.00","2,400.00","-100.00",-1.50%,-0.30%
"""
        filepath = tmp_path / "holdings.csv"
        filepath.write_text(csv_content)
        df = parse_holdings_csv(str(filepath))
        assert len(df) == 2
        assert "name" in df.columns
        assert df["units"].iloc[0] == 100.0

    def test_numeric_conversion(self, tmp_path):
        csv_content = """Instrument,Qty.,Avg. cost,LTP,Invested,Cur. val,P&L,Net chg.,Day chg.
"HDFC Fund",100.00,25.00,27.00,"2,500.00","2,700.00","200.00",2.00%,0.50%
"""
        filepath = tmp_path / "holdings.csv"
        filepath.write_text(csv_content)
        df = parse_holdings_csv(str(filepath))
        assert df["invested"].iloc[0] == 2500.0

    def test_single_holding(self, tmp_path):
        csv_content = """Instrument,Qty.,Avg. cost,LTP,Invested,Cur. val,P&L,Net chg.,Day chg.
"SBI Gold Fund",200.00,10.00,11.00,"2,000.00","2,200.00","200.00",5.00%,0.50%
"""
        filepath = tmp_path / "holdings.csv"
        filepath.write_text(csv_content)
        df = parse_holdings_csv(str(filepath))
        assert len(df) == 1
        assert df["units"].iloc[0] == 200.0


class TestComputeXirr:
    def test_positive_return(self):
        cfs = [(-1000, 0), (1100, 365)]
        xirr = compute_xirr(cfs)
        assert xirr > 0

    def test_negative_return(self):
        cfs = [(-1000, 0), (900, 365)]
        xirr = compute_xirr(cfs)
        assert xirr < 0

    def test_zero_return(self):
        cfs = [(-1000, 0), (1000, 365)]
        xirr = compute_xirr(cfs)
        assert abs(xirr) < 5

    def test_single_cf_returns_zero(self):
        cfs = [(-1000, 0)]
        xirr = compute_xirr(cfs)
        assert xirr == 0.0

    def test_empty_list_returns_zero(self):
        xirr = compute_xirr([])
        assert xirr == 0.0

    def test_multi_year(self):
        cfs = [(-10000, 0), (5000, 365), (6000, 730)]
        xirr = compute_xirr(cfs)
        assert isinstance(xirr, float)


class TestComputePortfolioStats:
    def test_empty_df(self):
        result = compute_portfolio_stats(pd.DataFrame())
        assert result == {}

    def test_total_invested(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert result["total_invested"] == 8000.0

    def test_total_current(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert result["total_current"] == 8300.0

    def test_total_pnl(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert result["total_pnl"] == 300.0

    def test_n_holdings(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert result["n_holdings"] == 3

    def test_holdings_list(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert len(result["holdings"]) == 3

    def test_holding_has_required_keys(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        h = result["holdings"][0]
        for key in ["name", "units", "invested", "current_value", "pnl", "return_pct", "weight"]:
            assert key in h

    def test_categories(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        assert "categories" in result
        assert len(result["categories"]) > 0

    def test_holdings_sorted_by_value(self, sample_holdings_df):
        result = compute_portfolio_stats(sample_holdings_df)
        values = [h["current_value"] for h in result["holdings"]]
        assert values == sorted(values, reverse=True)


class TestIndianMfMap:
    def test_has_known_funds(self):
        assert "HDFC Flexi Cap Fund" in INDIAN_MF_MAP
        assert "UTI Nifty 50 Index Fund" in INDIAN_MF_MAP

    def test_none_entries(self):
        none_funds = [k for k, v in INDIAN_MF_MAP.items() if v is None]
        assert len(none_funds) > 0

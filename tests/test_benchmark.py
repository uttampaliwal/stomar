"""Tests for paper vs Nifty benchmark (P3.2)."""

import os
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.signals.benchmarks import paper_vs_nifty, _max_drawdown_from_curve
from src.trading.ledger import Ledger


@pytest.fixture
def ledger():
    db_path = os.path.join(tempfile.mkdtemp(), "test_bench.db")
    lg = Ledger(db_path)
    yield lg
    lg.close()


def _price_df(start="2025-01-01", n=60, drift=0.002):
    dates = pd.bdate_range(start, periods=n)
    base = 100 * (1 + drift) ** np.arange(n)
    return pd.DataFrame({
        "open": base,
        "high": base * 1.01,
        "low": base * 0.99,
        "close": base,
        "volume": np.full(n, 1_000_000.0),
    }, index=dates)


class TestMaxDrawdown:
    def test_flat_curve(self):
        assert _max_drawdown_from_curve([100, 100, 100]) == 0.0

    def test_peak_to_trough(self):
        assert _max_drawdown_from_curve([100, 120, 90, 95]) == pytest.approx(0.25)

    def test_rising_curve_no_drawdown(self):
        assert _max_drawdown_from_curve([100, 110, 125]) == 0.0


class TestPaperVsNifty:
    def test_requires_snapshots(self, ledger):
        result = paper_vs_nifty(ledger)
        assert "error" in result

    def test_benchmark_curve_shapes(self, ledger):
        dates = pd.bdate_range("2025-03-01", periods=30)
        value = 100_000.0
        for d in dates:
            value *= 1.002
            ledger.log_snapshot(str(d.date()), round(value, 2), 5000, {})

        with patch("src.data.data_fetcher.fetch_stock_data",
                   return_value=_price_df(start="2025-01-01", n=120, drift=0.001)):
            result = paper_vs_nifty(ledger)

        assert "error" not in result, result
        assert result["paper"]["dates"][0] == "2025-03-03"
        assert len(result["paper"]["cumulative"]) == 30
        assert len(result["nifty"]["cumulative"]) >= 30
        # Paper drift 0.2%/day > nifty 0.1%/day → positive alpha
        assert result["alpha"] > 0
        assert result["paper_total_return"] > result["nifty_total_return"]
        assert result["information_ratio"] is not None
        assert result["n_snapshots"] == 30

    def test_paper_lagging_nifty_gives_negative_alpha(self, ledger):
        dates = pd.bdate_range("2025-03-01", periods=30)
        value = 100_000.0
        for d in dates:
            value *= 1.0005  # half of nifty drift
            ledger.log_snapshot(str(d.date()), round(value, 2), 5000, {})

        with patch("src.data.data_fetcher.fetch_stock_data",
                   return_value=_price_df(start="2025-01-01", n=120, drift=0.001)):
            result = paper_vs_nifty(ledger)

        assert result["alpha"] < 0

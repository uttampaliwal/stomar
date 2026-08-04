"""Tests for validated data fetching with auto-repair (P2.1/P2.2/P2.4)."""

import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import data_fetcher
from src.data.data_fetcher import fetch_validated_stock_data


@pytest.fixture
def tmp_data(tmp_path, monkeypatch):
    monkeypatch.setattr(data_fetcher, "DATA_DIR", str(tmp_path))
    monkeypatch.setattr(data_fetcher, "_MONITORING_DIR", str(tmp_path / "monitoring"))
    monkeypatch.setattr(data_fetcher, "_DATA_FIXES_PATH", str(tmp_path / "monitoring" / "data_fixes.json"))
    monkeypatch.setattr(data_fetcher, "_FETCH_STATS_PATH", str(tmp_path / "monitoring" / "fetch_stats.json"))
    return tmp_path


def _price_df(n=260, drop_days=()):
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    dates = dates.drop([d for d in drop_days if d in dates])
    close = 100 + np.cumsum(np.random.default_rng(1).normal(0, 0.3, len(dates)))
    return pd.DataFrame({
        "open": close, "high": close + 1, "low": close - 1,
        "close": close, "volume": np.full(len(dates), 1_000_000.0),
    }, index=dates)


class TestGapFilling:
    def test_extended_refetch_clears_gaps(self, tmp_data):
        # First fetch has a missing week; extended fetch is complete
        gap_days = pd.bdate_range(end=pd.Timestamp.today().normalize() - pd.Timedelta(days=40), periods=5)
        df_gappy = _price_df(n=200, drop_days=gap_days)
        df_full = _price_df(n=260)

        fetch_calls = {"n": 0}
        def fake_fetch(ticker, period="2y", interval="1d", force_refresh=False):
            fetch_calls["n"] += 1
            fetch_calls["period"] = period
            return df_full if force_refresh else df_gappy

        with patch.object(data_fetcher, "fetch_stock_data", side_effect=fake_fetch):
            result = fetch_validated_stock_data("TEST.NS", period="2y")

        assert result["df"] is df_full
        assert "gap_fill" in result["fixes"]
        # Repair logged to data_fixes.json
        fixes = json.loads((tmp_data / "monitoring" / "data_fixes.json").read_text())
        assert fixes[-1]["kind"] == "gap_fill"
        assert fixes[-1]["ticker"] == "TEST.NS"

    def test_holiday_gaps_are_not_treated_as_missing(self, tmp_data):
        # Diwali 2025 (Tue Oct 21) missing from an otherwise complete series
        df = _price_df(n=260)
        df = df.drop([pd.Timestamp("2025-10-21")])
        with patch.object(data_fetcher, "fetch_stock_data", return_value=df):
            result = fetch_validated_stock_data("TEST.NS", period="2y")
        assert "gap_fill" not in result["fixes"]


class TestCorporateActionRefetch:
    def test_suspect_move_triggers_refetch_and_clears(self, tmp_data):
        df_bad = _price_df(n=260)
        # Inject a fake -50% split crash (unadjusted price artifact)
        crash_pos = int(len(df_bad) * 0.4)
        df_bad.iloc[crash_pos:, df_bad.columns.get_loc("close")] *= 0.5
        df_good = _price_df(n=260)

        calls = {"n": 0}
        def fake_fetch(ticker, period="2y", interval="1d", force_refresh=False):
            calls["n"] += 1
            return df_bad if calls["n"] == 1 else df_good

        with patch.object(data_fetcher, "fetch_stock_data", side_effect=fake_fetch):
            result = fetch_validated_stock_data("TEST.NS", period="2y")

        assert "corporate_action" in result["fixes"]
        assert result["validation"].get("corporate_actions") == []


class TestFetchStats:
    def test_success_tracked(self, tmp_data):
        df = _price_df(n=200)
        with patch.object(data_fetcher, "fetch_stock_data", return_value=df):
            fetch_validated_stock_data("TEST.NS", period="1y")
        stats = json.loads((tmp_data / "monitoring" / "fetch_stats.json").read_text())
        assert stats["TEST.NS"]["successes"] == 1
        assert stats["TEST.NS"]["success_rate"] == 1.0

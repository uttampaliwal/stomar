"""Tests for src/data_validation.py."""

import numpy as np
import pandas as pd

from src.data_validation import (
    detect_gaps,
    detect_corporate_actions,
    detect_stale_data,
    validate_prices,
    validate_data,
)


def _make_df(n=100, start="2024-01-01", include_volume=True):
    """Create clean OHLCV DataFrame."""
    dates = pd.bdate_range(start, periods=n)
    np.random.seed(42)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    df = pd.DataFrame({
        "open": close - np.random.rand(n) * 0.5,
        "high": close + np.random.rand(n) * 1.0,
        "low": close - np.random.rand(n) * 1.0,
        "close": close,
        "volume": np.random.randint(1000, 10000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


# ── detect_gaps ──

class TestDetectGaps:
    def test_no_gaps(self):
        df = _make_df(50)
        assert detect_gaps(df) == []

    def test_single_gap(self):
        df = _make_df(50)
        # Remove days 20-24
        df = df.drop(df.index[20:25])
        gaps = detect_gaps(df)
        assert len(gaps) >= 1
        assert any(g["n_days"] >= 4 for g in gaps)

    def test_multiple_gaps(self):
        df = _make_df(100)
        df = df.drop(df.index[10:13])
        df = df.drop(df.index[50:54])
        gaps = detect_gaps(df)
        assert len(gaps) >= 2

    def test_empty_df(self):
        df = _make_df(0)
        assert detect_gaps(df) == []

    def test_short_df(self):
        df = _make_df(2)
        assert detect_gaps(df) == []


# ── detect_corporate_actions ──

class TestDetectCorporateActions:
    def test_clean_data_no_actions(self):
        df = _make_df(50)
        actions = detect_corporate_actions(df)
        assert actions == []

    def test_detects_split(self):
        df = _make_df(50)
        # Simulate 2:1 split: price halves, volume doubles
        idx = 25
        df.iloc[idx, df.columns.get_loc("close")] *= 0.5
        df.iloc[idx, df.columns.get_loc("open")] *= 0.5
        df.iloc[idx, df.columns.get_loc("high")] *= 0.5
        df.iloc[idx, df.columns.get_loc("low")] *= 0.5
        df.iloc[idx, df.columns.get_loc("volume")] *= 2.0
        actions = detect_corporate_actions(df, threshold=0.15)
        assert len(actions) >= 1

    def test_returns_dict_fields(self):
        df = _make_df(50)
        idx = 25
        df.iloc[idx, df.columns.get_loc("close")] *= 0.5
        df.iloc[idx, df.columns.get_loc("open")] *= 0.5
        df.iloc[idx, df.columns.get_loc("high")] *= 0.5
        df.iloc[idx, df.columns.get_loc("low")] *= 0.5
        actions = detect_corporate_actions(df, threshold=0.15)
        if actions:
            assert "date" in actions[0]
            assert "type" in actions[0]
            assert "return_pct" in actions[0]

    def test_empty_df(self):
        df = _make_df(0)
        assert detect_corporate_actions(df) == []


# ── detect_stale_data ──

class TestDetectStaleData:
    def test_fresh_data(self):
        dates = pd.bdate_range(end=pd.Timestamp.now(), periods=5)
        df = pd.DataFrame({"close": [100, 101, 102, 103, 104]}, index=dates)
        result = detect_stale_data(df, max_age_days=30)
        assert result["is_stale"] is False

    def test_stale_data(self):
        df = pd.DataFrame({
            "close": [100.0],
        }, index=pd.DatetimeIndex([pd.Timestamp.now() - pd.Timedelta(days=10)]))
        result = detect_stale_data(df, max_age_days=3)
        assert result["is_stale"] is True
        assert result["age_days"] == 10

    def test_empty_df(self):
        df = _make_df(0)
        result = detect_stale_data(df)
        assert result["is_stale"] is True

    def test_returns_all_keys(self):
        df = _make_df(5)
        result = detect_stale_data(df)
        assert "is_stale" in result
        assert "last_date" in result
        assert "age_days" in result
        assert "max_age_days" in result


# ── validate_prices ──

class TestValidatePrices:
    def test_clean_data_passes(self):
        df = _make_df(50)
        errors = validate_prices(df)
        assert errors == []

    def test_negative_prices(self):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("close")] = -5.0
        errors = validate_prices(df)
        assert any("non-positive" in e for e in errors)

    def test_zero_prices(self):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("open")] = 0.0
        errors = validate_prices(df)
        assert any("non-positive" in e for e in errors)

    def test_high_less_than_low(self):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("high")] = 50.0
        df.iloc[10, df.columns.get_loc("low")] = 150.0
        errors = validate_prices(df)
        assert any("high < low" in e for e in errors)

    def test_negative_volume(self):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("volume")] = -100.0
        errors = validate_prices(df)
        assert any("negative volume" in e for e in errors)

    def test_extreme_move(self):
        df = _make_df(50)
        df.iloc[10, df.columns.get_loc("close")] = df.iloc[9]["close"] * 2.0
        errors = validate_prices(df)
        assert any("extreme" in e for e in errors)

    def test_missing_column(self):
        df = pd.DataFrame({"close": [100, 101]})
        errors = validate_prices(df)
        assert any("Missing column" in e for e in errors)


# ── validate_data (full pipeline) ──

class TestValidateData:
    def test_clean_data_passes(self):
        df = _make_df(100)
        result = validate_data(df, "TEST.NS")
        assert result["passed"] is True
        assert result["errors"] == []
        assert result["data_points"] == 100

    def test_bad_data_fails(self):
        df = _make_df(100)
        df.iloc[10, df.columns.get_loc("close")] = -5.0
        result = validate_data(df, "TEST.NS")
        assert result["passed"] is False
        assert len(result["errors"]) > 0

    def test_returns_all_fields(self):
        df = _make_df(10)
        result = validate_data(df, "TEST.NS")
        assert "ticker" in result
        assert "passed" in result
        assert "errors" in result
        assert "warnings" in result
        assert "gaps" in result
        assert "corporate_actions" in result
        assert "stale_info" in result
        assert "data_points" in result
        assert "date_range" in result

    def test_ticker_preserved(self):
        df = _make_df(10)
        result = validate_data(df, "RELIANCE.NS")
        assert result["ticker"] == "RELIANCE.NS"

    def test_stale_warning(self):
        df = pd.DataFrame({
            "open": [100.0], "high": [101.0], "low": [99.0], "close": [100.5],
            "volume": [5000.0],
        }, index=pd.DatetimeIndex([pd.Timestamp.now() - pd.Timedelta(days=10)]))
        result = validate_data(df, "TEST.NS")
        assert any("old" in w for w in result["warnings"])

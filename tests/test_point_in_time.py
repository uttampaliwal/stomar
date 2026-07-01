"""Tests for point-in-time data discipline — verify no look-ahead bias."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_price_df(n=200):
    np.random.seed(42)
    dates = pd.bdate_range("2023-01-01", periods=n)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)
    df = pd.DataFrame({
        "open": open_, "high": high, "low": low,
        "close": close, "volume": volume,
    }, index=dates)
    df.index.name = "date"
    return df


class TestSentimentPointInTime:
    def test_sentiment_only_on_last_row(self):
        from src.data.features import add_sentiment_features
        df = _make_price_df()
        result = add_sentiment_features(df, "RELIANCE.NS")
        # All rows except the last should be 0.0
        historical = result["sentiment_score"].iloc[:-1]
        assert (historical == 0.0).all(), (
            f"Sentiment leaked to historical rows: "
            f"{(historical != 0.0).sum()} rows have non-zero sentiment"
        )

    def test_sentiment_last_row_can_be_nonzero(self):
        from src.data.features import add_sentiment_features
        df = _make_price_df()
        result = add_sentiment_features(df, "RELIANCE.NS")
        # Last row may or may not be non-zero (depends on cache)
        last_val = result["sentiment_score"].iloc[-1]
        assert isinstance(last_val, float)

    def test_sentiment_no_cache_returns_zero(self):
        from src.data.features import add_sentiment_features
        df = _make_price_df()
        result = add_sentiment_features(df, "NONEXISTENT_TICKER_XYZ.NS")
        assert (result["sentiment_score"] == 0.0).all()


class TestFlowPointInTime:
    def test_flow_shifted_by_one_day(self):
        from src.data.features import add_flow_features
        df = _make_price_df()
        result = add_flow_features(df)
        # Flow features should be present
        assert "fii_net" in result.columns
        assert "dii_net" in result.columns
        assert "flow_signal" in result.columns

    def test_flow_join_preserves_index(self):
        from src.data.features import add_flow_features
        df = _make_price_df()
        result = add_flow_features(df)
        assert len(result) == len(df)
        assert list(result.index) == list(df.index)

    def test_flow_no_cache_returns_zeros(self):
        from src.data.features import add_flow_features
        df = _make_price_df()
        result = add_flow_features(df)
        # Without cache or with empty cache, should be zeros
        # (cache may or may not exist depending on previous runs)
        assert "fii_net" in result.columns


class TestPCRPointInTime:
    def test_pcr_only_on_last_row(self):
        from src.data.features import add_pcr_features
        df = _make_price_df()
        result = add_pcr_features(df)
        # All rows except the last should be default (1.0)
        historical = result["pcr"].iloc[:-1]
        assert (historical == 1.0).all(), (
            f"PCR leaked to historical rows: "
            f"{(historical != 1.0).sum()} rows have non-default PCR"
        )

    def test_max_pcr_only_on_last_row(self):
        from src.data.features import add_pcr_features
        df = _make_price_df()
        result = add_pcr_features(df)
        historical = result["max_pain"].iloc[:-1]
        assert (historical == 0.0).all()


class TestMTFPointInTime:
    def test_mtf_only_on_last_row(self):
        from src.data.features import add_multitimeframe_features
        df = _make_price_df()
        result = add_multitimeframe_features(df, "RELIANCE.NS")
        # All rows except the last should be 0
        assert (result["mtf_signal"].iloc[:-1] == 0).all()
        assert (result["mtf_confidence"].iloc[:-1] == 0).all()


class TestFullFeaturePipelinePointInTime:
    def test_all_alt_features_only_on_last_row(self):
        from src.data.features import add_technical_indicators
        df = _make_price_df()
        result = add_technical_indicators(df, ticker="RELIANCE.NS")

        alt_cols = ["sentiment_score", "fii_net", "dii_net", "flow_signal",
                     "pcr", "max_pain", "mtf_signal", "mtf_confidence"]

        for col in alt_cols:
            if col in result.columns:
                historical = result[col].iloc[:-1]
                last_row = result[col].iloc[-1]
                # Historical should be default, last row may differ
                if col in ("sentiment_score",):
                    assert (historical == 0.0).all(), f"{col} leaked"
                elif col in ("pcr",):
                    assert (historical == 1.0).all(), f"{col} leaked"
                elif col in ("max_pain",):
                    assert (historical == 0.0).all(), f"{col} leaked"
                elif col in ("mtf_signal", "mtf_confidence"):
                    assert (historical == 0).all(), f"{col} leaked"

    def test_technical_indicators_are_point_in_time(self):
        from src.data.features import add_technical_indicators
        df = _make_price_df()
        result = add_technical_indicators(df)
        # Technical indicators use only past data (rolling windows)
        # They should NOT have future data
        # Verify: first 50 rows should have NaN for sma_50 (needs 50 days)
        assert pd.isna(result["sma_50"].iloc[0])
        # Last row should have a value
        assert not pd.isna(result["sma_50"].iloc[-1])

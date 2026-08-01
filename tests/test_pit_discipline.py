"""Point-in-time discipline: deterministic tests against real cache data.

The weak spot of the earlier suite is that it exercised the feature
functions with whatever happened to be in ``data/``. These tests write
their OWN cache files into a temp dir (via monkeypatched DATA_DIR) so the
mechanics that prevent look-ahead bias are verified exactly:

* sentiment score for day T lands ONLY on the last row — historical rows
  must stay 0.0 because that news was not yet observable;
* FII/DII flow for day T is published after T's close, so it joins to
  T+1's price (shift-by-one-day), never to T itself;
* ``ichimoku_chikou`` deliberately holds FUTURE closes (close.shift(-26)).
  The assertion below locks in that exact semantic so it remains a
  documented informational column — test_safety_leakage.py pins that it
  never reaches a model input.
"""

import json

import numpy as np
import pandas as pd
import pytest

import src.data.features as features_mod


@pytest.fixture
def empty_data_dir(tmp_path, monkeypatch):
    """Redirect feature caches to a fresh temp dir."""
    monkeypatch.setattr(features_mod, "DATA_DIR", str(tmp_path))
    return tmp_path


def _price_df(n=30):
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2025-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "open": close + rng.normal(0, 0.1, n),
        "high": close + np.abs(rng.normal(0, 0.3, n)) + 0.1,
        "low": close - np.abs(rng.normal(0, 0.3, n)) - 0.1,
        "close": close,
        "volume": rng.integers(100_000, 1_000_000, n),
    }, index=dates)


class TestSentimentPointInTime:
    def test_score_lands_only_on_last_row_with_real_cache(self, empty_data_dir):
        (empty_data_dir / "sentiment_RELIANCE_NS.json").write_text(
            json.dumps({"score": 0.42})
        )
        df = _price_df()
        result = features_mod.add_sentiment_features(df, "RELIANCE.NS")
        assert (result["sentiment_score"].iloc[:-1] == 0.0).all(), (
            "sentiment leaked to historical rows"
        )
        assert result["sentiment_score"].iloc[-1] == 0.42

    def test_missing_cache_leaves_all_rows_zero(self, empty_data_dir):
        result = features_mod.add_sentiment_features(_price_df(), "RELIANCE.NS")
        assert (result["sentiment_score"] == 0.0).all()

    def test_score_value_comes_from_cache_not_price(self, empty_data_dir):
        (empty_data_dir / "sentiment_RELIANCE_NS.json").write_text(
            json.dumps({"score": -0.75})
        )
        result = features_mod.add_sentiment_features(_price_df(), "RELIANCE.NS")
        assert result["sentiment_score"].iloc[-1] == -0.75


class TestFlowShiftByOneDay:
    def test_flow_for_T_lands_on_next_trading_day(self, empty_data_dir):
        df = _price_df(n=30)
        flow_date = df.index[10]
        pd.DataFrame({
            "date": [flow_date],
            "fii_net": [500.0],
            "dii_net": [300.0],
        }).to_parquet(empty_data_dir / "fii_dii.parquet")

        result = features_mod.add_flow_features(df)

        # day T itself must NOT see its own flow (published after T's close)
        assert result.loc[flow_date, "fii_net"] == 0.0
        assert result.loc[flow_date, "dii_net"] == 0.0
        # the NEXT trading day must carry T's flow
        next_date = df.index[11]
        assert result.loc[next_date, "fii_net"] == 500.0
        assert result.loc[next_date, "dii_net"] == 300.0
        # every other row untouched
        others = result["fii_net"].drop(index=[flow_date, next_date])
        assert (others == 0.0).all()

    def test_flow_signal_signs_follow_shift(self, empty_data_dir):
        df = _price_df(n=30)
        d1, d2, d3 = df.index[5], df.index[6], df.index[7]
        pd.DataFrame({
            "date": [d1, d2, d3],
            "fii_net": [500.0, -600.0, 100.0],
            "dii_net": [300.0, 100.0, -200.0],
        }).to_parquet(empty_data_dir / "fii_dii.parquet")

        result = features_mod.add_flow_features(df)

        # both positive -> +1 ; fii < -500 -> -1 ; otherwise neutral
        assert result.loc[df.index[6], "flow_signal"] == 1.0
        assert result.loc[df.index[7], "flow_signal"] == -1.0
        assert result.loc[df.index[8], "flow_signal"] == 0.0
        # flow rows themselves are still neutral
        assert result.loc[d1, "flow_signal"] == 0.0

    def test_missing_cache_yields_zero_columns(self, empty_data_dir):
        result = features_mod.add_flow_features(_price_df())
        assert (result["fii_net"] == 0.0).all()
        assert (result["dii_net"] == 0.0).all()
        assert (result["flow_signal"] == 0.0).all()

    def test_row_count_and_index_preserved(self, empty_data_dir):
        df = _price_df(n=30)
        result = features_mod.add_flow_features(df)
        assert len(result) == len(df)
        assert list(result.index) == list(df.index)


class TestIchimokuChikouSemantics:
    def test_chikou_holds_future_close_by_design(self):
        from src.signals.feature_pipeline import add_ichimoku

        df = _price_df(n=80)
        out = add_ichimoku(df.copy())

        # row i carries close[i+26] — a FUTURE close, only informational
        np.testing.assert_allclose(
            out["ichimoku_chikou"].iloc[:-26].to_numpy(),
            df["close"].iloc[26:].to_numpy(),
        )
        # the last 26 rows have no future close to show
        assert out["ichimoku_chikou"].iloc[-26:].isna().all()

    def test_other_ichimoku_components_use_only_past_data(self):
        from src.signals.feature_pipeline import add_ichimoku

        df = _price_df(n=80)
        out = add_ichimoku(df.copy())

        # tenkan/kijun at row i use high/low up to i (no negative shifts)
        expected_tenkan = (df["high"].rolling(9).max() + df["low"].rolling(9).min()) / 2
        np.testing.assert_allclose(out["ichimoku_tenkan"].to_numpy(),
                                   expected_tenkan.to_numpy(), equal_nan=True)

        # senkou spans are the (shifted +26) midpoint lines — lag, never lead
        expected_a = ((expected_tenkan + out["ichimoku_kijun"]) / 2).shift(26)
        np.testing.assert_allclose(out["ichimoku_senkou_a"].to_numpy(),
                                   expected_a.to_numpy(), equal_nan=True)

    def test_chikou_never_reaches_model_schema(self):
        from src.models.trainer import FEATURE_COLS

        assert "ichimoku_chikou" not in FEATURE_COLS

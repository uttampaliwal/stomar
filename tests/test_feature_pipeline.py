"""Tests for src/signals/feature_pipeline.py — SOTA factor pipeline (76 factors)."""

import numpy as np
import pandas as pd
import pytest

from src.signals.feature_pipeline import (
    FACTOR_GROUPS,
    TARGET_COLUMNS,
    add_cmf,
    add_fii_momentum,
    add_high_low_spread,
    add_ichimoku,
    add_order_flow_imbalance,
    add_pcr_slope,
    add_relative_strength,
    add_supertrend,
    add_vol_adjusted_momentum,
    add_volatility_zscore,
    all_factor_names,
    compute_features,
    factor_count,
    load_index_history,
)


@pytest.fixture
def ohlcv():
    rng = np.random.default_rng(7)
    n = 400
    close = 100 * np.exp(np.cumsum(rng.normal(0.0005, 0.01, n)))
    high = close * (1 + rng.uniform(0, 0.01, n))
    low = close * (1 - rng.uniform(0, 0.01, n))
    open_ = low + rng.uniform(0, 1, n) * (high - low)
    volume = rng.integers(100_000, 1_000_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
    )


def _trending(down: bool = False):
    rng = np.random.default_rng(3)
    n = 300
    drift = -0.002 if down else 0.002
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.004, n)))
    high = close * (1 + 0.005)
    low = close * (1 - 0.005)
    open_ = low + (high - low) / 2
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": np.full(n, 500_000)},
        index=pd.date_range("2023-01-02", periods=n, freq="B"),
    )


class TestFactorRegistry:
    def test_factor_count(self):
        assert factor_count() >= 60
        assert len(all_factor_names()) == factor_count()

    def test_factor_names_unique(self):
        names = all_factor_names()
        assert len(set(names)) == len(names)

    def test_factor_groups_cover_names(self):
        flat = set(all_factor_names())
        grouped = {f for group in FACTOR_GROUPS.values() for f in group}
        assert flat == grouped

    def test_groups_contain_sota_factors(self):
        assert "supertrend" in FACTOR_GROUPS["trend"]
        assert "ichimoku_tenkan" in FACTOR_GROUPS["trend"]
        assert "mom_12m_voladj" in FACTOR_GROUPS["momentum"]
        assert "rel_strength_3m" in FACTOR_GROUPS["momentum"]
        assert "vol_zscore" in FACTOR_GROUPS["volatility"]
        assert "cmf" in FACTOR_GROUPS["volume"]
        assert "ofi" in FACTOR_GROUPS["microstructure"]
        assert "pcr_slope" in FACTOR_GROUPS["microstructure"]
        assert "fii_momentum" in FACTOR_GROUPS["microstructure"]
        assert "high_low_spread" in FACTOR_GROUPS["microstructure"]


class TestAddSupertrend:
    def test_uptrend_below_close(self):
        df = add_supertrend(_trending(down=False))
        up = df[df["supertrend_dir"] == 1]
        assert len(up) > 50
        assert (up["supertrend"] <= up["close"]).all()

    def test_downtrend_above_close(self):
        df = add_supertrend(_trending(down=True))
        dn = df[df["supertrend_dir"] == 0]
        assert len(dn) > 50
        assert (dn["supertrend"] >= dn["close"]).all()

    def test_direction_binary(self):
        df = add_supertrend(_trending())
        assert set(df["supertrend_dir"].dropna().unique()) <= {0.0, 1.0}

    def test_default_columns(self):
        df = add_supertrend(_trending())
        assert {"supertrend", "supertrend_dir"} <= set(df.columns)


class TestAddIchimoku:
    def test_columns_present(self, ohlcv):
        df = add_ichimoku(ohlcv.copy())
        for c in ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
                  "ichimoku_senkou_b", "ichimoku_chikou"]:
            assert c in df.columns

    def test_senkou_a_is_shifted_midpoint(self, ohlcv):
        df = add_ichimoku(ohlcv.copy())
        mid = (df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2
        np.testing.assert_allclose(
            df["ichimoku_senkou_a"].to_numpy()[26:],
            mid.to_numpy()[:-26],
            equal_nan=True,
        )

    def test_senkou_b_is_shifted_52_hl_midpoint(self, ohlcv):
        df = add_ichimoku(ohlcv.copy())
        mid52 = (ohlcv["high"].rolling(52).max() + ohlcv["low"].rolling(52).min()) / 2
        np.testing.assert_allclose(
            df["ichimoku_senkou_b"].to_numpy()[26:],
            mid52.to_numpy()[:-26],
            equal_nan=True,
        )

    def test_chikou_is_leading_close(self, ohlcv):
        df = add_ichimoku(ohlcv.copy())
        np.testing.assert_allclose(
            df["ichimoku_chikou"].to_numpy()[:-26],
            df["close"].to_numpy()[26:],
            equal_nan=True,
        )


class TestAddCmf:
    def test_bounded(self, ohlcv):
        cmf = add_cmf(ohlcv.copy())["cmf"].dropna()
        assert ((cmf >= -1.0) & (cmf <= 1.0)).all()

    def test_positive_when_price_rises_on_volume(self):
        rng = np.random.default_rng(1)
        n = 200
        close = 100 + np.arange(n) * 0.2
        open_ = close - 0.1
        high = np.maximum(open_, close) * 1.005
        low = np.minimum(open_, close) * 0.995
        vol = rng.integers(100_000, 200_000, n)
        df = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close, "volume": vol})
        cmf = add_cmf(df, period=20)["cmf"].dropna()
        assert (cmf > 0).mean() > 0.8


class TestAddOrderFlowImbalance:
    def test_column_present(self, ohlcv):
        df = add_order_flow_imbalance(ohlcv.copy(), period=10)
        assert "ofi" in df.columns

    def test_mean_near_zero_on_balanced_data(self):
        rng = np.random.default_rng(2)
        n = 500
        close = 100 + np.cumsum(rng.normal(0, 0.05, n))
        open_ = close + rng.normal(0, 0.05, n)
        high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.005, n))
        low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.005, n))
        df = pd.DataFrame({"open": open_, "high": high, "low": low,
                           "close": close,
                           "volume": rng.integers(100_000, 900_000, n)})
        ofi = add_order_flow_imbalance(df, period=10)["ofi"].dropna()
        assert abs(ofi.mean()) < 0.05


class TestAddVolatilityZscore:
    def test_column_present(self, ohlcv):
        df = add_volatility_zscore(ohlcv.copy(), period=20)
        assert "vol_zscore" in df.columns

    def test_roughly_unit_scale(self, ohlcv):
        df = add_volatility_zscore(ohlcv.copy(), period=20)
        z = df["vol_zscore"].dropna()
        assert abs(z.std()) < 5


class TestAddVolAdjustedMomentum:
    def test_columns_present(self, ohlcv):
        df = add_vol_adjusted_momentum(ohlcv.copy())
        for c in ["mom_1m_voladj", "mom_3m_voladj", "mom_6m_voladj", "mom_12m_voladj"]:
            assert c in df.columns

    def test_positive_on_uptrend(self):
        df = add_vol_adjusted_momentum(_trending(down=False))
        assert df["mom_1m_voladj"].iloc[-1] > 0

    def test_negative_on_downtrend(self):
        df = add_vol_adjusted_momentum(_trending(down=True))
        assert df["mom_1m_voladj"].iloc[-1] < 0


class TestAddRelativeStrength:
    def test_nan_without_index(self, ohlcv):
        df = add_relative_strength(ohlcv.copy(), index_df=None)
        assert df["rel_strength_1m"].isna().all()

    def test_with_index(self, ohlcv):
        index_df = ohlcv[["close"]].copy()
        df = add_relative_strength(ohlcv.copy(), index_df=index_df)
        vals = df["rel_strength_1m"].dropna()
        assert len(vals) > 0
        assert vals.iloc[-1] == pytest.approx(0.0, abs=1e-9)

    def test_columns_present(self, ohlcv):
        df = add_relative_strength(ohlcv.copy(), index_df=ohlcv[["close"]])
        assert {"rel_strength_1m", "rel_strength_3m"} <= set(df.columns)


class TestAddHighLowSpread:
    def test_column_present(self, ohlcv):
        df = add_high_low_spread(ohlcv.copy(), period=20)
        assert "high_low_spread" in df.columns

    def test_nonnegative(self, ohlcv):
        df = add_high_low_spread(ohlcv.copy(), period=20)
        assert (df["high_low_spread"].dropna() >= 0).all()


class TestPitExternal:
    def test_pcr_slope_only_last_row(self, ohlcv):
        df = add_pcr_slope(ohlcv.copy(), period=5)
        assert "pcr_slope" in df.columns
        assert df["pcr_slope"].iloc[:-1].fillna(0.0).eq(0.0).all()
        assert pd.notna(df["pcr_slope"].iloc[-1])

    def test_fii_momentum_only_last_row(self, ohlcv):
        df = add_fii_momentum(ohlcv.copy(), period=5)
        assert "fii_momentum" in df.columns
        assert df["fii_momentum"].iloc[:-1].fillna(0.0).eq(0.0).all()
        assert pd.notna(df["fii_momentum"].iloc[-1])


class TestComputeFeatures:
    def test_raises_on_empty(self):
        with pytest.raises(ValueError):
            compute_features(pd.DataFrame())

    def test_full_factor_set_without_external(self, ohlcv):
        df = compute_features(ohlcv.copy(), include_external=False)
        present = [f for f in all_factor_names() if f in df.columns]
        assert len(present) >= 60
        assert {"supertrend", "cmf", "ofi", "vol_zscore",
                "mom_1m_voladj", "rel_strength_1m"} <= set(df.columns)

    def test_targets_present(self, ohlcv):
        df = compute_features(ohlcv.copy(), include_external=False)
        assert set(TARGET_COLUMNS) <= set(df.columns)
        assert df["target_direction"].dropna().isin([0, 1]).all()

    def test_external_skipped_without_ticker(self, ohlcv):
        df = compute_features(ohlcv.copy(), ticker=None)
        for c in ["sentiment_score", "fii_net", "pcr", "mtf_signal", "pcr_slope"]:
            assert c not in df.columns

    def test_external_called_with_ticker(self, ohlcv, monkeypatch):
        called = {}

        def fake_add(df, ticker=None, **kw):
            called[ticker] = called.get(ticker, 0) + 1
            return df

        for name in ["add_sentiment_features", "add_flow_features", "add_pcr_features",
                     "add_multitimeframe_features"]:
            monkeypatch.setattr(
                f"src.signals.feature_pipeline.{name}", fake_add, raising=False
            )
        monkeypatch.setattr("src.signals.feature_pipeline.add_pcr_slope",
                            lambda df, period=5: df)
        monkeypatch.setattr("src.signals.feature_pipeline.add_fii_momentum",
                            lambda df, period=5: df)
        df = compute_features(ohlcv.copy(), ticker="TEST.NS")
        assert called.get("TEST.NS", 0) >= 1

    def test_no_infinite_values(self, ohlcv):
        df = compute_features(ohlcv.copy(), include_external=False)
        assert not np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).any()

    def test_preserves_index(self, ohlcv):
        df = compute_features(ohlcv.copy(), include_external=False)
        pd.testing.assert_index_equal(df.index, ohlcv.index)


class TestLoadIndexHistory:
    def test_returns_none_when_all_sources_fail(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no store")

        monkeypatch.setattr("src.data.store.get_store", boom)
        import sys
        fake_yf = type("YF", (), {"download": lambda *a, **k: pd.DataFrame()})()
        monkeypatch.setitem(sys.modules, "yfinance", fake_yf)
        assert load_index_history(period="1y") is None

    def test_returns_close_columns_from_store(self, monkeypatch):
        idx = pd.DataFrame(
            {"close": np.arange(100.0, 130.0)},
            index=pd.date_range("2024-01-01", periods=30, freq="B"),
        )

        class FakeStore:
            def get_history(self, symbol, adjusted=False):
                return idx.copy()

        monkeypatch.setattr("src.data.store.get_store", lambda: FakeStore())
        out = load_index_history(period="1y")
        assert out is not None
        assert list(out.columns) == ["close"]

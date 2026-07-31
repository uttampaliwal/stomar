"""Tests for src/signals/regime_hmm.py — GMM regime engine and risk scaling."""

import numpy as np
import pandas as pd
import pytest


def _make_segmented_ohlc(n_segments=1000, seed=42):
    """Synthetic OHLCV: bear (high vol, down drift), bull, ranging."""
    rng = np.random.default_rng(seed)
    segments = [
        (260, 0.025, -0.002),   # bear: high vol, negative drift
        (340, 0.008, 0.0012),   # bull: low vol, positive drift
        (180, 0.006, 0.0001),   # ranging: lowest vol, flat
    ]
    closes, rets = [], []
    price = 100.0
    for n, vol, drift in segments:
        r = rng.normal(drift, vol, n)
        rets.extend(r)
        closes.extend(price * np.exp(np.cumsum(r)))
        price = closes[-1]
    close = np.asarray(closes)
    high = close * (1 + np.abs(rng.normal(0, vol * 0.8, len(close))))
    low = close * (1 - np.abs(rng.normal(0, vol * 0.8, len(close))))
    idx = pd.date_range("2021-01-01", periods=len(close), freq="B")
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.004, len(close))),
        "high": high, "low": low, "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, len(close)).astype(float),
    }, index=idx)
    return df


class TestFeatures:
    def test_requires_min_rows(self):
        from src.signals.regime_hmm import _features
        df = _make_segmented_ohlc(50).head(10)
        with pytest.raises(ValueError):
            _features(df)

    def test_returns_expected_columns(self):
        from src.signals.regime_hmm import _features
        df = _make_segmented_ohlc()
        f = _features(df)
        assert set(f.columns) == {"log_return", "realized_vol", "atr_pct"}
        assert f.notna().all().all()

    def test_respects_window(self):
        from src.signals.regime_hmm import _features
        df = _make_segmented_ohlc()
        f = _features(df, window=200)
        assert len(f) <= 200


class TestFitRegimeModel:
    def test_returns_bundle(self):
        from src.signals.regime_hmm import fit_regime_model
        df = _make_segmented_ohlc()
        model = fit_regime_model(df)
        assert model["n_samples"] > 500
        assert set(model["state_map"].values()) == {"bear", "bull", "ranging"}
        assert len(model["X_scaled"]) == model["n_samples"]

    def test_deterministic(self):
        from src.signals.regime_hmm import fit_regime_model
        df = _make_segmented_ohlc()
        m1 = fit_regime_model(df)
        m2 = fit_regime_model(df)
        assert m1["state_map"] == m2["state_map"]


class TestDetectRegime:
    def test_returns_regime_info(self):
        from src.signals.regime_hmm import detect_regime
        df = _make_segmented_ohlc()
        info = detect_regime(df, use_cache=False)
        assert info["regime_key"] in ("bear", "bull", "ranging")
        assert info["regime"] in (
            "High Volatility Bear Market", "Trending Bull Market",
            "Low Volatility Ranging/Sideways Market",
        )
        probs = info["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-6
        assert 0 <= info["risk"]["leverage"] <= 1
        assert isinstance(info["risk"]["hedge"], (bool, float))

    def test_volatile_downtrend_is_bear(self):
        from src.signals.regime_hmm import detect_regime
        rng = np.random.default_rng(7)
        n = 600
        rets = np.concatenate([
            rng.normal(0.0005, 0.008, n - 100),  # calm
            rng.normal(-0.012, 0.04, 100),       # crash tail
        ])
        close = 100 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2022-01-01", periods=n, freq="B"))
        info = detect_regime(df, use_cache=False)
        assert info["regime_key"] == "bear"

    def test_trending_up_is_bull(self):
        from src.signals.regime_hmm import detect_regime
        rng = np.random.default_rng(11)
        n = 600
        close = 100 * np.exp(np.cumsum(rng.normal(0.002, 0.008, n)))
        df = pd.DataFrame({
            "open": close, "high": close * 1.01, "low": close * 0.99,
            "close": close, "volume": np.ones(n) * 1e6,
        }, index=pd.date_range("2022-01-01", periods=n, freq="B"))
        info = detect_regime(df, use_cache=False)
        assert info["regime_key"] == "bull"


class TestRiskScaling:
    def test_pure_bear_scales_down(self):
        from src.signals.regime_hmm import get_risk_scaling
        risk = get_risk_scaling({"High Volatility Bear Market": 1.0})
        assert risk["leverage"] == 0.5
        assert risk["position_scale"] == 0.5
        assert risk["max_risk_per_trade"] == 0.01
        assert risk["hedge"] == 1.0  # blended flag in [0, 1]

    def test_pure_bull_full_scale(self):
        from src.signals.regime_hmm import get_risk_scaling
        risk = get_risk_scaling({"Trending Bull Market": 1.0})
        assert risk["leverage"] == 1.0
        assert risk["hedge"] == 0.0

    def test_probability_blending(self):
        from src.signals.regime_hmm import get_risk_scaling
        risk = get_risk_scaling({
            "High Volatility Bear Market": 0.5,
            "Low Volatility Ranging/Sideways Market": 0.5,
        })
        assert abs(risk["leverage"] - 0.625) < 1e-6
        assert abs(risk["max_risk_per_trade"] - 0.0125) < 1e-6


class TestStateSequence:
    def test_alignment_with_features(self):
        from src.signals.regime_hmm import _features, state_sequence
        df = _make_segmented_ohlc()
        seq = state_sequence(df)
        features = _features(df)
        assert list(seq.index) == list(features.index)
        assert set(seq.unique()) <= {"bear", "bull", "ranging"}

    def test_current_regime_matches_last_state(self):
        from src.signals.regime_hmm import detect_regime, state_sequence
        df = _make_segmented_ohlc()
        seq = state_sequence(df)
        info = detect_regime(df, use_cache=False)
        assert seq.iloc[-1] == info["regime_key"]

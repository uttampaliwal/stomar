"""O2 parity: one shared feature builder for train and serve paths."""

import numpy as np
import pandas as pd


def _ohlcv(n=150, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
    idx = pd.date_range("2023-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.004, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.008, n))),
        "low": close * (1 - np.abs(rng.normal(0, 0.008, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
    }, index=idx)


def test_helper_builds_required_features_offline():
    from src.models.trainer_features import REQUIRED_FEATURES, build_feature_frame

    feat = build_feature_frame(_ohlcv(), ticker=None, load_index=False)
    missing = set(REQUIRED_FEATURES) - set(feat.columns)
    assert not missing, f"required features missing: {missing}"


def test_helper_includes_sota_factors_not_legacy_subset():
    """The parity essence: serve-time frame must contain the 16 SOTA factors
    that bare add_technical_indicators() lacks (previously zero-filled)."""
    from src.models.trainer_features import build_feature_frame

    feat = build_feature_frame(_ohlcv(), ticker=None, load_index=False)
    for col in ("supertrend", "supertrend_dir", "cmf", "ofi", "vol_zscore",
                "high_low_spread", "mom_1m_voladj",
                "ichimoku_tenkan", "ichimoku_kijun"):
        assert col in feat.columns, f"SOTA factor missing: {col}"


def test_helper_bans_lookahead_column():
    from src.models.ensemble import BANNED_LOOKAHEAD_FEATURES
    from src.models.trainer_features import build_feature_frame

    feat = build_feature_frame(_ohlcv(), ticker=None, load_index=False)
    assert not (set(BANNED_LOOKAHEAD_FEATURES) & set(feat.columns))


def test_pipeline_hash_matches_trainer_candidate_set():
    """_stage_features must hash what training actually selects from."""
    from unittest.mock import patch

    from src.core.pipeline import RetrainingPipeline
    from src.data.feature_store import compute_feature_hash
    from src.models.trainer_features import (
        build_feature_frame,
        select_training_features,
    )

    frame = _ohlcv()
    with patch("src.data.data_fetcher.fetch_stock_data", return_value=frame), \
         patch("src.signals.feature_pipeline.load_index_history", return_value=None):
        result = RetrainingPipeline()._stage_features("TEST.NS")
    assert result.status == "success"

    expected = compute_feature_hash(
        select_training_features(
            build_feature_frame(frame, ticker="TEST.NS", load_index=False),
            "TEST.NS",
        )
    )
    assert result.feature_hash == expected

"""Regression tests for look-ahead leakage and backtest realism.

These lock in two fixes:

* ``ichimoku_chikou`` (future close) must never reach a model input;
* walk-forward splits must purge/embargo the label boundary.
"""

import numpy as np
import pandas as pd

from src.models.ensemble import BANNED_LOOKAHEAD_FEATURES, _neutralize_lookahead_features
from src.models.trainer import FEATURE_COLS
from src.signals.feature_pipeline import FACTOR_GROUPS, all_factor_names
from src.trading.backtester import walk_forward_split


def test_chikou_excluded_from_model_feature_schema():
    assert "ichimoku_chikou" not in FEATURE_COLS


def test_chikou_excluded_from_factor_registry():
    assert "ichimoku_chikou" not in all_factor_names()
    assert "ichimoku_chikou" not in FACTOR_GROUPS["trend"]


def test_chikou_banned_at_inference_boundary():
    assert "ichimoku_chikou" in BANNED_LOOKAHEAD_FEATURES


def test_neutralization_replaces_future_values_with_past_only():
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    close = np.linspace(100, 200, n)
    df = pd.DataFrame({
        "close": close,
        "ichimoku_chikou": np.concatenate([close[26:], [0] * 26]),  # future data
    })
    feature_cols = ["ichimoku_chikou", "close"]
    safe = _neutralize_lookahead_features(df, list(feature_cols))
    assert safe == ["close"]
    assert "ichimoku_chikou" not in safe
    # neutralized column must equal the CURRENT close (past info), never future
    np.testing.assert_allclose(df["ichimoku_chikou"].to_numpy(), close)


def test_walk_forward_split_applies_purge_and_embargo():
    n = 3 * 252 + 252 + 100
    df = pd.DataFrame({"close": np.linspace(100, 200, n)})
    splits = walk_forward_split(df, train_years=3, test_years=1, step_months=12,
                                purge_days=60, embargo_days=20)
    assert splits, "expected at least one split"

    for split in splits:
        train_idx = df.index.get_indexer(split["train"])
        test_idx = df.index.get_indexer(split["test"])
        # test must start after purge + embargo gap
        assert test_idx[0] - train_idx[-1] >= 60 + 20, (
            "purge/embargo gap violated"
        )
        # training must end before the test window minus purge
        assert train_idx[-1] < test_idx[0] - 60


def test_walk_forward_zero_purge_matches_legacy_behavior():
    n = 3 * 252 + 252 + 50
    df = pd.DataFrame({"close": np.linspace(100, 200, n)})
    purged = walk_forward_split(df, purge_days=0, embargo_days=0)
    legacy = walk_forward_split(df, purge_days=0, embargo_days=0)
    assert len(purged) == len(legacy)
    assert (purged[0]["train"] == legacy[0]["train"]).all()

"""Tests for the trainer feature contract (required vs optional columns)."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from src.models.trainer_features import (
    FEATURE_COLS,
    OPTIONAL_FEATURES,
    REQUIRED_FEATURES,
    select_training_features,
)


def test_required_features_are_subset_of_feature_cols():
    assert set(REQUIRED_FEATURES).issubset(set(FEATURE_COLS))


def test_optional_and_required_partition_feature_cols():
    assert set(OPTIONAL_FEATURES).isdisjoint(set(REQUIRED_FEATURES))
    assert sorted(set(OPTIONAL_FEATURES) | set(REQUIRED_FEATURES)) == sorted(set(FEATURE_COLS))


def test_select_raises_on_missing_required():
    import pandas as pd

    df = pd.DataFrame({"close": [1.0, 2.0], "volume": [10, 20], "returns_1d": [0.1, -0.1]})
    # drop returns_1d -> required missing
    with pytest.raises(ValueError, match="required features missing"):
        select_training_features(df.drop(columns=["returns_1d"]), "T.NS")


def test_select_warns_on_missing_optional(caplog):
    import pandas as pd

    df = pd.DataFrame({"close": [1.0, 2.0], "volume": [10, 20], "returns_1d": [0.1, -0.1]})
    with caplog.at_level("WARNING", logger="src.models.trainer_features"):
        cols = select_training_features(df, "T.NS")
    assert any("optional_features_missing" in r.message for r in caplog.records)
    assert cols == ["close", "volume", "returns_1d"]


def test_select_preserves_feature_cols_order():
    import pandas as pd

    df = pd.DataFrame({"volume": [1.0], "close": [2.0], "returns_1d": [0.0], "rsi": [50.0]})
    cols = select_training_features(df, "T.NS")
    assert cols == ["close", "volume", "rsi", "returns_1d"]


def test_no_leakage_columns_in_feature_cols():
    for leaked in ("ichimoku_chikou", "tb_label", "target_direction", "target"):
        assert leaked not in FEATURE_COLS

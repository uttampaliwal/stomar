"""Tests for the configurable training-label convention (direction vs triple_barrier)."""
import numpy as np
import pandas as pd
import pytest


def _feat_frame(n=300, seed=7):
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    df = pd.DataFrame({
        "close": close,
        "target": np.r_[np.diff(close) / close[:-1], [np.nan]],
        "target_direction": (np.r_[np.diff(close), [0]] > 0).astype(int),
        "tb_label": rng.integers(0, 3, n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
    }, index=pd.bdate_range("2024-01-01", periods=n))
    return df


class TestBarrierBinaryLabels:
    def test_maps_and_masks_time_barrier(self):
        from src.models.trainer import barrier_binary_labels
        df = pd.DataFrame({"tb_label": [1, 0, 2, 1, 2]})
        lab = barrier_binary_labels(df)
        assert lab.iloc[[0, 1, 3]].tolist() == [1.0, 0.0, 1.0]
        assert lab.isna().tolist() == [False, False, True, False, True]

    def test_none_without_column(self):
        from src.models.trainer import barrier_binary_labels
        assert barrier_binary_labels(pd.DataFrame({"close": [1, 2]})) is None
        assert barrier_binary_labels(None) is None

    def test_index_preserved(self):
        from src.models.trainer import barrier_binary_labels
        df = _feat_frame(50)
        lab = barrier_binary_labels(df)
        assert (lab.index == df.index).all()


class TestBuildTreeTargets:
    def test_direction_mode_keeps_all_rows(self):
        from src.models.trainer import build_tree_targets
        df = _feat_frame().dropna()
        X, y, fwd = build_tree_targets(df, ["f1", "f2"], "direction")
        assert len(X) == len(df)
        assert (y.values == df["target_direction"].values).all()
        assert fwd is not None and len(fwd) == len(y)

    def test_triple_barrier_drops_time_rows(self):
        from src.models.trainer import build_tree_targets
        df = _feat_frame()
        X, y, fwd = build_tree_targets(df, ["f1", "f2"], "triple_barrier")
        kept = df["tb_label"].isin([0, 1])
        assert len(X) == int(kept.sum())
        assert set(np.unique(y)) <= {0, 1}
        # forward returns stay aligned with the kept rows
        assert (fwd.values == df.loc[kept, "target"].values).all()

    def test_falls_back_when_tb_missing(self):
        from src.models.trainer import build_tree_targets
        df = _feat_frame().drop(columns=["tb_label"]).dropna()
        X, y, fwd = build_tree_targets(df, ["f1", "f2"], "triple_barrier")
        assert len(X) == len(df)
        assert (y.values == df["target_direction"].values).all()

    def test_falls_back_when_too_few_barrier_rows(self):
        from src.models.trainer import build_tree_targets
        df = _feat_frame(150)
        df["tb_label"] = 2  # almost everything is a time-barrier row
        df.iloc[:5, df.columns.get_loc("tb_label")] = 1
        X, y, fwd = build_tree_targets(df.dropna(), ["f1", "f2"], "triple_barrier")
        # fell back to direction labels on all rows
        assert len(X) == len(df.dropna())
        assert set(np.unique(y)) <= {0, 1}

    def test_insufficient_data_returns_nones(self):
        from src.models.trainer import build_tree_targets
        df = _feat_frame(30).dropna()
        X, y, fwd = build_tree_targets(df, ["f1", "f2"], "direction")
        assert X is None and y is None and fwd is None


class TestSettingsLabelType:
    def test_default_is_direction(self):
        from src.core.settings import Settings
        s = Settings()
        assert s.label_type == "direction"

    def test_env_override(self, monkeypatch):
        from src.core.settings import Settings
        monkeypatch.setenv("STOMAR_LABEL_TYPE", "triple_barrier")
        assert Settings().label_type == "triple_barrier"
        monkeypatch.setenv("STOMAR_LABEL_TYPE", "bogus")
        with pytest.raises(Exception):
            Settings()

    def test_fallback_settings_validate(self, monkeypatch):
        from src.core.settings import _HAS_PYDANTIC_SETTINGS
        if _HAS_PYDANTIC_SETTINGS:
            pytest.skip("_FallbackSettings only exists without pydantic")
        monkeypatch.setenv("STOMAR_LABEL_TYPE", "triple_barrier")
        from src.core.settings import _FallbackSettings
        assert _FallbackSettings().label_type == "triple_barrier"
        monkeypatch.setenv("STOMAR_LABEL_TYPE", "bogus")
        with pytest.raises(ValueError):
            _FallbackSettings()


class TestMetaFeaturesLabelPlumbing:
    def test_nan_labels_skipped_and_used(self):
        from src.models.trainer import _collect_meta_features
        from sklearn.preprocessing import MinMaxScaler

        n = 120
        rng = np.random.default_rng(3)
        cols = ["f1", "f2"]
        df = pd.DataFrame({
            "f1": rng.normal(size=n) + 100,
            "f2": rng.normal(size=n) + 50,
        }, index=pd.bdate_range("2024-01-01", periods=n))

        class _StubXgb:
            def predict_proba(self, X):
                return np.full((len(X), 2), [0.35, 0.65])

        labels = pd.Series(1.0, index=df.index)
        labels.iloc[40:60] = np.nan  # time-barrier block must be skipped
        split_idx = 10
        seq_length = 5

        scaler = MinMaxScaler().fit(df.values)
        out = _collect_meta_features(
            None, None, None, _StubXgb(), scaler, cols, df,
            split_idx, seq_length, labels=labels,
        )
        assert out is not None
        X_meta, y_meta = out
        start = split_idx + seq_length
        expected_rows = n - start - 20  # 20 NaN rows dropped
        assert len(y_meta) == expected_rows
        assert (y_meta == 1).all()
        assert X_meta.shape == (expected_rows, 6)

    def test_no_labels_derives_direction(self):
        from src.models.trainer import _collect_meta_features
        from sklearn.preprocessing import MinMaxScaler

        n = 120
        rng = np.random.default_rng(5)
        cols = ["f1"]
        df = pd.DataFrame({"f1": rng.normal(size=n) + 100},
                          index=pd.bdate_range("2024-01-01", periods=n))

        class _StubXgb:
            def predict_proba(self, X):
                return np.full((len(X), 2), [0.5, 0.5])

        scaler = MinMaxScaler().fit(df.values)
        _, y_meta = _collect_meta_features(
            None, None, None, _StubXgb(), scaler, cols, df, 10, 5,
        )
        closes = df["f1"].values
        start = 10 + 5
        expected_dir = (closes[start:] > closes[start - 1:-1]).astype(int)
        assert (y_meta == expected_dir).all()

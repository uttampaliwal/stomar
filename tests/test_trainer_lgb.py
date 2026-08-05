"""Tests for src/trainer.py — LightGBM training call conventions."""

import numpy as np
import pandas as pd

from src.models.trainer import train_for_ticker


class _FakeTree:
    """Sklearn-API stand-in for xgb/lgb/catboost models."""

    def __init__(self):
        self.fit_kwargs = None
        self._n = 0

    def fit(self, *args, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict(self, X):
        self._n += 1
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        self._n += 1
        return np.column_stack([np.zeros(len(X)), np.ones(len(X))])


class TestLightGBMFitCall:
    def test_lgb_fit_uses_eval_set(self, monkeypatch):
        n = 120
        df = pd.DataFrame({
            "f1": np.linspace(0, 1, n),
            "f2": np.random.RandomState(0).rand(n),
            "target_direction": [0, 1] * (n // 2),
            "target": np.random.RandomState(1).rand(n),
        })

        monkeypatch.setattr("src.models.trainer.fetch_stock_data",
                            lambda ticker, period="5y", force_refresh=False: df.copy())
        monkeypatch.setattr("src.signals.feature_pipeline.compute_features",
                            lambda df, ticker=None, index_df=None: df)
        monkeypatch.setattr("src.signals.feature_pipeline.load_index_history",
                            lambda period="5y": None)
        monkeypatch.setattr("src.models.trainer.select_training_features",
                            lambda df, ticker: ["f1", "f2"])
        monkeypatch.setattr("src.models.trainer._walk_forward_dl",
                            lambda scaled, seq_length, n_splits=5: (None, []))
        monkeypatch.setattr("src.models.trainer.save_models",
                            lambda *a, **k: None)
        monkeypatch.setattr("src.models.trainer._collect_meta_features",
                            lambda *a, **k: None)

        X_tr = df[["f1", "f2"]].iloc[:80]
        y_tr = df["target_direction"].iloc[:80]
        X_te = df[["f1", "f2"]].iloc[80:].copy()
        y_te = df["target_direction"].iloc[80:].copy()

        def fake_walk_forward(X, y, ticker, returns=None, n_splits=5, embargo=5, horizon=1):
            X_te.attrs["oos_metrics"] = {}
            return X_tr, X_te, y_tr, y_te, []

        monkeypatch.setattr("src.models.trainer._walk_forward_xgb", fake_walk_forward)

        lgb = _FakeTree()
        monkeypatch.setattr("src.models.trainer.build_lgb_model", lambda: lgb)
        monkeypatch.setattr("src.models.trainer.build_xgb_model", _FakeTree)
        monkeypatch.setattr("src.models.trainer.build_catboost_model", _FakeTree)

        train_for_ticker("TEST.NS")

        assert lgb.fit_kwargs is not None
        eval_set = lgb.fit_kwargs.get("eval_set")
        assert eval_set is not None
        assert len(eval_set) == 1
        x, y = eval_set[0]
        assert x.equals(X_te)
        assert y.equals(y_te)
        assert "eval_X" not in lgb.fit_kwargs
        assert "eval_y" not in lgb.fit_kwargs

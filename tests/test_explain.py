"""Tests for src/signals/interpretability.py — SHAP explanations."""

import numpy as np
import pandas as pd
import pytest

from src.signals.interpretability import (
    _build_summary,
    _xgb_pred_contribs,
    explain_prediction,
    explain_prediction_shap,
    feature_label,
)


@pytest.fixture
def trained_xgb():
    import xgboost as xgb

    rng = np.random.default_rng(0)
    n = 300
    X = pd.DataFrame(
        {"rsi": rng.normal(0, 1, n), "macd": rng.normal(0, 1, n),
         "sma_20": rng.normal(0, 1, n), "volume_ratio": rng.normal(0, 1, n)}
    )
    y = (X["rsi"] + 2 * X["macd"] + 0.5 * X["volume_ratio"] > 0).astype(int)
    assert y.nunique() == 2, "fixture must produce both classes"
    model = xgb.XGBClassifier(n_estimators=20, max_depth=3, random_state=42)
    model.fit(X, y)
    return model, list(X.columns)


@pytest.fixture
def features(trained_xgb):
    _, cols = trained_xgb
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {"rsi": rng.normal(0, 1, 100), "macd": rng.normal(0, 1, 100),
         "sma_20": rng.normal(0, 1, 100), "volume_ratio": rng.normal(0, 1, 100)}
    )
    return df, cols


class TestFeatureLabel:
    def test_known_mapping(self):
        assert feature_label("rsi") == "RSI_14"
        assert feature_label("supertrend") == "Supertrend"
        assert feature_label("ichimoku_tenkan") == "Ichimoku tenkan-sen"

    def test_unknown_fallback(self):
        assert feature_label("custom_feature") == "Custom Feature"


class TestXgbPredContribs:
    def test_shape(self, trained_xgb):
        model, cols = trained_xgb
        X = pd.DataFrame({c: [1.0] for c in cols})
        contribs = _xgb_pred_contribs(model, X)
        assert contribs.shape == (1, len(cols) + 1)


class TestExplainPredictionShap:
    def test_shap_explanation(self, trained_xgb, features, monkeypatch):
        model, cols = trained_xgb
        df, _ = features
        tuple_like = (None, None, None, model, None, cols, None)

        def fake_models_exist(ticker):
            return True

        def fake_load_models(ticker):
            return tuple_like

        monkeypatch.setattr("src.models.model.models_exist",
                            fake_models_exist)
        monkeypatch.setattr("src.models.model.load_models",
                            fake_load_models)

        out = explain_prediction_shap("TEST.NS", df, cols)
        assert out["ticker"] == "TEST.NS"
        assert out["method"] == "shap"
        assert out["signal"] in ("BUY", "SELL")
        assert 0.0 <= out["probability_up"] <= 1.0
        assert len(out["top_features"]) == min(10, len(cols))
        for c in out["top_features"]:
            assert "feature" in c
            assert "pct_contribution" in c
            assert 0 <= c["pct_contribution"] <= 100
        assert len(out["summary"]) > 0
        assert out["total_features"] == len(cols)

    def test_pct_contributions_sum_approx_100(self, trained_xgb, features, monkeypatch):
        model, cols = trained_xgb
        df, _ = features
        monkeypatch.setattr("src.models.model.models_exist",
                            lambda t: True)
        monkeypatch.setattr("src.models.model.load_models",
                            lambda t: (None, None, None, model, None, cols, None))
        out = explain_prediction_shap("TEST.NS", df, cols)
        total = sum(c["pct_contribution"] for c in out["top_features"])
        assert total == pytest.approx(100.0, abs=1.0)

    def test_no_models_error(self, monkeypatch):
        monkeypatch.setattr("src.models.model.models_exist",
                            lambda t: False)
        out = explain_prediction_shap("NOPE.NS", pd.DataFrame(), ["a"])
        assert "error" in out

    def test_summary_direction_matches_shap_sign(self):
        latest = pd.DataFrame({"rsi": [70.0]})
        contribs = [
            {"feature": "rsi", "shap_value": 0.5, "pct_contribution": 60.0},
            {"feature": "macd", "shap_value": -0.2, "pct_contribution": 40.0},
        ]
        summary = _build_summary(contribs, latest, "BUY", 0.0, "shap")
        assert any("RSI_14" in s and "bullish" in s and "60.0%" in s for s in summary)
        assert any("MACD" in s and "bearish" in s for s in summary)
        assert summary[0] == "Model signals BUY based on the most recent bar."


class TestExplainPredictionFallback:
    def test_falls_back_to_gain_on_shap_failure(self, trained_xgb, features, monkeypatch):
        model, cols = trained_xgb
        df, _ = features

        def broken_shap(*a, **k):
            raise RuntimeError("shap exploded")

        monkeypatch.setattr("src.signals.interpretability.explain_prediction_shap",
                            broken_shap)
        monkeypatch.setattr("src.models.model.models_exist",
                            lambda t: True)
        monkeypatch.setattr("src.models.model.load_models",
                            lambda t: (None, None, None, model, None, cols, None))

        out = explain_prediction("TEST.NS", df, cols)
        assert out["method"] == "gain"
        assert len(out["top_features"]) > 0
        assert all(c["pct_contribution"] >= 0 for c in out["top_features"])

    def test_error_when_no_model(self):
        out = explain_prediction("NOPE.NS", pd.DataFrame(), ["a"])
        assert "error" in out

    def test_gain_importance_ordering(self, trained_xgb, monkeypatch):
        model, cols = trained_xgb
        rng = np.random.default_rng(2)
        df = pd.DataFrame({c: rng.normal(size=50) for c in cols})
        monkeypatch.setattr("src.models.model.models_exist",
                            lambda t: True)
        monkeypatch.setattr("src.models.model.load_models",
                            lambda t: (None, None, None, model, None, cols, None))
        out = explain_prediction_shap("TEST.NS", df, cols)
        pcts = [c["pct_contribution"] for c in out["top_features"]]
        assert pcts == sorted(pcts, reverse=True)


class TestModelColumnAlignment:
    def test_uses_model_feature_names(self, trained_xgb, features, monkeypatch):
        """Model trained on fewer columns than the feature matrix -> align."""
        model, cols = trained_xgb
        df, _ = features
        # add extra columns not in the model
        df = df.assign(extra_feat=np.arange(len(df)), other_feat=np.arange(len(df)))

        monkeypatch.setattr("src.models.model.models_exist", lambda t: True)
        monkeypatch.setattr("src.models.model.load_models",
                            lambda t: (None, None, None, model, None, cols, None))
        out = explain_prediction_shap("TEST.NS", df, cols + ["extra_feat", "other_feat"])
        assert out["total_features"] == len(cols)
        assert not any(c["feature"] in ("extra_feat", "other_feat")
                       for c in out["top_features"])

    def test_all_nan_columns_dropped(self, trained_xgb, features, monkeypatch):
        model, cols = trained_xgb
        df, _ = features
        df["rel_strength_1m"] = np.nan
        monkeypatch.setattr("src.models.model.models_exist", lambda t: True)
        monkeypatch.setattr("src.models.model.load_models",
                            lambda t: (None, None, None, model, None, cols, None))
        out = explain_prediction_shap("TEST.NS", df, cols + ["rel_strength_1m"])
        assert "error" not in out
        assert out["total_features"] == len(cols)
        assert not any(c["feature"] == "rel_strength_1m" for c in out["top_features"])

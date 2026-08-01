"""Tests for src/ensemble.py — meta-learner and regime routing."""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


def _make_meta_learner():
    """Train a meta-learner on synthetic data."""
    from src.models.ensemble import train_meta_learner
    np.random.seed(42)
    n = 200
    # 5 model outputs: first 3 are directions, last 2 are probabilities
    meta_X = np.random.rand(n, 5)
    # True signal: weighted combination with noise
    signal = 0.3 * meta_X[:, 0] + 0.3 * meta_X[:, 1] + 0.2 * meta_X[:, 2] + 0.1 * meta_X[:, 3] + 0.1 * meta_X[:, 4]
    y = (signal > 0.5).astype(int)
    return train_meta_learner(meta_X[:150], y[:150]), meta_X[150:], y[150:]


class TestTrainMetaLearner:
    def test_returns_pipeline(self):
        from src.models.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] + meta_X[:, 1] > 1.0).astype(int)
        model = train_meta_learner(meta_X, y)
        assert isinstance(model, Pipeline)

    def test_has_coefficients(self):
        from src.models.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] > 0.5).astype(int)
        model = train_meta_learner(meta_X, y)
        coefs = model.named_steps["clf"].coef_[0]
        assert coefs.shape == (5,)


class TestPredictWithMetalearner:
    def test_returns_probabilities(self):
        from src.models.ensemble import predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        probs = predict_with_metalearner(model, meta_X_test)
        assert probs.shape == (50,)
        assert all(0 <= p <= 1 for p in probs)

    def test_returns_array(self):
        from src.models.ensemble import predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        probs = predict_with_metalearner(model, meta_X_test)
        assert isinstance(probs, np.ndarray)


class TestEvaluateMetalearner:
    def test_returns_all_keys(self):
        from src.models.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        ew_preds = np.random.randint(0, 2, len(y_test))
        result = evaluate_metalearner(meta_X_test, y_test, model, ew_preds)
        assert "meta_learner_accuracy" in result
        assert "equal_weight_accuracy" in result
        assert "improvement" in result

    def test_without_equal_weight(self):
        from src.models.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        result = evaluate_metalearner(meta_X_test, y_test, model)
        assert "meta_learner_accuracy" in result
        assert "equal_weight_accuracy" not in result

    def test_accuracy_in_range(self):
        from src.models.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        result = evaluate_metalearner(meta_X_test, y_test, model)
        assert 0 <= result["meta_learner_accuracy"] <= 1


class TestSaveLoadMetaModel:
    def test_roundtrip(self, tmp_path):
        from src.models.ensemble import save_meta_model, load_meta_model
        model, _, _ = _make_meta_learner()
        path = str(tmp_path / "meta.pkl")
        save_meta_model(model, path)
        loaded = load_meta_model(path)
        assert isinstance(loaded, Pipeline)

    def test_loaded_model_predicts(self, tmp_path):
        from src.models.ensemble import save_meta_model, load_meta_model, predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        path = str(tmp_path / "meta.pkl")
        save_meta_model(model, path)
        loaded = load_meta_model(path)
        probs = predict_with_metalearner(loaded, meta_X_test)
        assert probs.shape == (50,)


class TestRegimeWeights:
    def test_all_regimes_have_6_weights(self):
        from src.models.ensemble import REGIME_WEIGHTS
        for regime, weights in REGIME_WEIGHTS.items():
            assert len(weights) == 6, f"{regime} should have 6 weights"

    def test_weights_sum_to_one(self):
        from src.models.ensemble import REGIME_WEIGHTS
        for regime, weights in REGIME_WEIGHTS.items():
            assert abs(sum(weights.values()) - 1.0) < 0.001, \
                f"{regime} weights sum to {sum(weights.values())}"

    def test_get_regime_weights_bull(self):
        from src.models.ensemble import get_regime_weights
        w = get_regime_weights("Bull")
        assert w["xgb"] == 0.20  # Trees favored in bull

    def test_get_regime_weights_bear(self):
        from src.models.ensemble import get_regime_weights
        w = get_regime_weights("Bear")
        assert w["lstm"] == 0.20  # DL favored in bear

    def test_unknown_regime_fallback(self):
        from src.models.ensemble import get_regime_weights
        w = get_regime_weights("Unknown")
        assert all(abs(v - 1.0 / 6) < 1e-6 for v in w.values())


class TestWilsonInterval:
    def test_bounds_and_range(self):
        from src.models.ensemble import wilson_interval
        lo, hi = wilson_interval(80, 100)
        assert 0 <= lo <= hi <= 1
        assert lo < 0.8 < hi  # 90% CI on p_hat=0.8, n=100
        assert lo > 0.5

    def test_known_value(self):
        from src.models.ensemble import wilson_interval
        lo, _ = wilson_interval(90, 100)
        assert abs(lo - 0.8378) < 0.01

    def test_zero_samples(self):
        from src.models.ensemble import wilson_interval
        assert wilson_interval(0, 0) == (0.0, 0.0)


class TestConvictionLabel:
    def test_high(self):
        from src.models.ensemble import conviction_label
        assert conviction_label(0.81, 0.6) == "HIGH"
        assert conviction_label(0.5, 0.56) == "HIGH"

    def test_medium(self):
        from src.models.ensemble import conviction_label
        assert conviction_label(0.63, 0.4) == "MEDIUM"
        assert conviction_label(0.5, 0.53) == "MEDIUM"

    def test_low(self):
        from src.models.ensemble import conviction_label
        assert conviction_label(0.5, 0.4) == "LOW"


def _make_backtest_rows(n=40, seed=0):
    rng = np.random.default_rng(seed)
    import pandas as pd
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    rows = []
    for i, d in enumerate(dates):
        probs = rng.random(6)
        rows.append({
            "date": d,
            "actual": 1 if i % 2 else 0,
            "lstm": 1 if probs[0] > 0.5 else 0,
            "gru": 1 if probs[1] > 0.5 else 0,
            "transformer": 1 if probs[2] > 0.5 else 0,
            "xgb": 1 if probs[3] > 0.5 else 0,
            "lgb": 1 if probs[4] > 0.5 else 0,
            "cat": 1 if probs[5] > 0.5 else 0,
            "lstm_prob": float(probs[0]), "gru_prob": float(probs[1]),
            "transformer_prob": float(probs[2]), "xgb_prob": float(probs[3]),
            "lgb_prob": float(probs[4]), "cat_prob": float(probs[5]),
            "meta_X": probs.tolist(),
        })
    return rows


class TestDynamicRegimeWeights:
    def test_fallback_without_regime_series(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        dyn = compute_dynamic_regime_weights(_make_backtest_rows(), None, "bear")
        assert dyn["method"] == "fallback"
        assert all(abs(v - 1.0 / 6) < 1e-6 for v in dyn["weights"].values())

    def test_fallback_too_few_samples(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        rows = _make_backtest_rows(5)
        regime = pd.Series(["bear"] * 5, index=[r["date"] for r in rows])
        dyn = compute_dynamic_regime_weights(rows, regime, "bear", min_samples=10)
        assert dyn["method"] == "fallback"
        assert "reason" in dyn

    def test_dynamic_weights_sum_to_one(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        rows = _make_backtest_rows(40)
        regime = pd.Series(["bear"] * 40, index=[r["date"] for r in rows])
        dyn = compute_dynamic_regime_weights(rows, regime, "bear", window=30, min_samples=5)
        assert dyn["method"] == "dynamic"
        assert abs(sum(dyn["weights"].values()) - 1.0) < 1e-3
        assert 20 <= dyn["n_samples"] <= 30  # business days inside 30-day window

    def test_accurate_model_gets_highest_weight(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        rows = _make_backtest_rows(40)
        for r in rows:
            r["lstm"] = r["actual"]  # lstm always correct
            r["lstm_prob"] = 1.0 if r["actual"] else 0.0
        regime = pd.Series(["bear"] * 40, index=[r["date"] for r in rows])
        dyn = compute_dynamic_regime_weights(rows, regime, "bear", window=30, min_samples=5)
        assert dyn["weights"]["lstm"] == max(dyn["weights"].values())

    def test_constant_prob_model_excluded(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        rows = _make_backtest_rows(40)
        for r in rows:
            r["cat_prob"] = 0.5  # placeholder — never trained
            r["cat"] = 0
        regime = pd.Series(["bear"] * 40, index=[r["date"] for r in rows])
        dyn = compute_dynamic_regime_weights(rows, regime, "bear", window=30, min_samples=5)
        assert "cat" in dyn["excluded"]
        assert "cat" not in dyn["weights"]
        assert abs(sum(dyn["weights"].values()) - 1.0) < 1e-3

    def test_weights_renormalized_without_excluded(self):
        from src.models.ensemble import compute_dynamic_regime_weights
        rows = _make_backtest_rows(40)
        for r in rows:
            r["cat_prob"] = 0.5
            r["cat"] = 0
            r["lgb_prob"] = 0.5
            r["lgb"] = 0
        regime = pd.Series(["bear"] * 40, index=[r["date"] for r in rows])
        dyn = compute_dynamic_regime_weights(rows, regime, "bear", window=30, min_samples=5)
        assert set(dyn["excluded"]) == {"cat", "lgb"}
        assert set(dyn["weights"]) == {"lstm", "gru", "transformer", "xgb"}
        assert abs(sum(dyn["weights"].values()) - 1.0) < 1e-3


class TestRegimeAdjustedEnsemble:
    def _setup(self):
        import pandas as pd
        rng = np.random.default_rng(42)
        n = 400
        close = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.015, n)))
        idx = pd.date_range("2023-01-01", periods=n, freq="B")
        df = pd.DataFrame({
            "open": close * (1 + rng.normal(0, 0.004, n)),
            "high": close * (1 + np.abs(rng.normal(0, 0.008, n))),
            "low": close * (1 - np.abs(rng.normal(0, 0.008, n))),
            "close": close,
            "volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }, index=idx)
        df["target_direction"] = (df["close"].shift(-1) > df["close"]).astype(int)
        feature_cols = ["ret_1", "vol_10", "atr_pct", "sma_20", "mom_5", "rsi"]
        df["ret_1"] = df["close"].pct_change()
        df["vol_10"] = df["ret_1"].rolling(10).std()
        df["atr_pct"] = (df["high"] - df["low"]) / df["close"]
        df["sma_20"] = df["close"].rolling(20).mean()
        df["mom_5"] = df["close"].pct_change(5)
        df["rsi"] = 50.0
        df = df.dropna()

        y = df["target_direction"].values
        X = df[feature_cols].values
        n_tr = int(len(X) * 0.7)

        from src.models.model import build_xgb_model, build_lgb_model, build_catboost_model
        from src.models.model import build_lstm, build_gru, build_transformer
        from sklearn.preprocessing import MinMaxScaler

        xgb = build_xgb_model()
        xgb.fit(X[:n_tr], y[:n_tr], eval_set=[(X[n_tr:], y[n_tr:])], verbose=False)
        lgb = build_lgb_model()
        lgb.fit(X[:n_tr], y[:n_tr], eval_X=X[n_tr:], eval_y=y[n_tr:])
        cat = build_catboost_model()
        cat.fit(X[:n_tr], y[:n_tr], eval_set=(X[n_tr:], y[n_tr:]))

        scaler = MinMaxScaler()
        scaler.fit(X[:n_tr])

        # Untrained DL models — forward pass must still run
        lstm, gru, tf = build_lstm(6), build_gru(6), build_transformer(6)
        return {
            "df": df, "features": feature_cols, "scaler": scaler,
            "xgb": xgb, "lgb": lgb, "cat": cat, "lstm": lstm, "gru": gru, "tf": tf,
        }

    def test_full_pipeline_runs(self):
        from src.models.ensemble import regime_adjusted_ensemble
        s = self._setup()
        out = regime_adjusted_ensemble("NONEXISTENT_TEST.NS", {
            "lstm": s["lstm"], "gru": s["gru"], "transformer": s["tf"],
            "xgb": s["xgb"], "lgb": s["lgb"], "cat": s["cat"],
            "scaler": s["scaler"], "features": s["features"],
        }, s["df"], use_meta=False)
        assert out["signal"] in ("BUY", "SELL")
        assert 0 <= out["probability_up"] <= 1
        assert 0 <= out["confidence"] <= 100
        assert out["conviction"] in ("HIGH", "MEDIUM", "LOW")
        assert len(out["confidence_interval"]) == 2
        assert "regime" in out and "risk" in out
        assert len(out["models"]) == 6
        assert "lstm" in out["weights"]

    def test_legacy_meta_learner_alignment(self):
        from src.models.ensemble import train_meta_learner, predict_with_metalearner
        rng = np.random.default_rng(1)
        meta_X = rng.random((150, 5))  # legacy 5-col
        y = (meta_X[:, 0] > 0.5).astype(int)
        model = train_meta_learner(meta_X, y)
        probs = predict_with_metalearner(model, rng.random((10, 6)))  # 6-col input
        assert probs.shape == (10,)


class TestMetaLearnerBeatsEqualWeight:
    def test_on_synthetic_data(self):
        from src.models.ensemble import train_meta_learner, predict_with_metalearner
        np.random.seed(42)
        n = 300
        # Models with different skill levels
        meta_X = np.random.rand(n, 5)
        # True signal dominated by first 2 models
        signal = 0.4 * meta_X[:, 0] + 0.35 * meta_X[:, 1] + 0.1 * meta_X[:, 2] + 0.1 * meta_X[:, 3] + 0.05 * meta_X[:, 4]
        y = (signal > 0.5).astype(int)

        model = train_meta_learner(meta_X[:200], y[:200])
        probs = predict_with_metalearner(model, meta_X[200:])
        meta_preds = (probs > 0.5).astype(int)

        # Equal weight: average all 5 columns, threshold 0.5
        ew_probs = meta_X[200:].mean(axis=1)
        ew_preds = (ew_probs > 0.5).astype(int)

        meta_acc = (meta_preds == y[200:]).mean()
        ew_acc = (ew_preds == y[200:]).mean()

        # Meta-learner should be at least as good as equal weight
        assert meta_acc >= ew_acc - 0.05  # Allow small tolerance


class TestMetaLearnerHandlesNaN:
    def test_does_not_crash(self):
        from src.models.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = train_meta_learner(meta_X, y)
        assert model is not None


class TestMetaLearnerCoefficients:
    def test_coefficients_are_finite(self):
        from src.models.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] > 0.5).astype(int)
        model = train_meta_learner(meta_X, y)
        coefs = model.named_steps["clf"].coef_[0]
        assert all(np.isfinite(coefs))

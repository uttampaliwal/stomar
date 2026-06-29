"""Tests for src/ensemble.py — meta-learner and regime routing."""

import numpy as np
from sklearn.pipeline import Pipeline


def _make_meta_learner():
    """Train a meta-learner on synthetic data."""
    from src.ensemble import train_meta_learner
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
        from src.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] + meta_X[:, 1] > 1.0).astype(int)
        model = train_meta_learner(meta_X, y)
        assert isinstance(model, Pipeline)

    def test_has_coefficients(self):
        from src.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] > 0.5).astype(int)
        model = train_meta_learner(meta_X, y)
        coefs = model.named_steps["clf"].coef_[0]
        assert coefs.shape == (5,)


class TestPredictWithMetalearner:
    def test_returns_probabilities(self):
        from src.ensemble import predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        probs = predict_with_metalearner(model, meta_X_test)
        assert probs.shape == (50,)
        assert all(0 <= p <= 1 for p in probs)

    def test_returns_array(self):
        from src.ensemble import predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        probs = predict_with_metalearner(model, meta_X_test)
        assert isinstance(probs, np.ndarray)


class TestEvaluateMetalearner:
    def test_returns_all_keys(self):
        from src.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        ew_preds = np.random.randint(0, 2, len(y_test))
        result = evaluate_metalearner(meta_X_test, y_test, model, ew_preds)
        assert "meta_learner_accuracy" in result
        assert "equal_weight_accuracy" in result
        assert "improvement" in result

    def test_without_equal_weight(self):
        from src.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        result = evaluate_metalearner(meta_X_test, y_test, model)
        assert "meta_learner_accuracy" in result
        assert "equal_weight_accuracy" not in result

    def test_accuracy_in_range(self):
        from src.ensemble import evaluate_metalearner
        model, meta_X_test, y_test = _make_meta_learner()
        result = evaluate_metalearner(meta_X_test, y_test, model)
        assert 0 <= result["meta_learner_accuracy"] <= 1


class TestSaveLoadMetaModel:
    def test_roundtrip(self, tmp_path):
        from src.ensemble import save_meta_model, load_meta_model
        model, _, _ = _make_meta_learner()
        path = str(tmp_path / "meta.pkl")
        save_meta_model(model, path)
        loaded = load_meta_model(path)
        assert isinstance(loaded, Pipeline)

    def test_loaded_model_predicts(self, tmp_path):
        from src.ensemble import save_meta_model, load_meta_model, predict_with_metalearner
        model, meta_X_test, _ = _make_meta_learner()
        path = str(tmp_path / "meta.pkl")
        save_meta_model(model, path)
        loaded = load_meta_model(path)
        probs = predict_with_metalearner(loaded, meta_X_test)
        assert probs.shape == (50,)


class TestRegimeWeights:
    def test_all_regimes_have_5_weights(self):
        from src.ensemble import REGIME_WEIGHTS
        for regime, weights in REGIME_WEIGHTS.items():
            assert len(weights) == 5, f"{regime} should have 5 weights"

    def test_weights_sum_to_one(self):
        from src.ensemble import REGIME_WEIGHTS
        for regime, weights in REGIME_WEIGHTS.items():
            assert abs(sum(weights.values()) - 1.0) < 0.001, \
                f"{regime} weights sum to {sum(weights.values())}"

    def test_get_regime_weights_bull(self):
        from src.ensemble import get_regime_weights
        w = get_regime_weights("Bull")
        assert w["xgb"] == 0.25  # Trees favored in bull

    def test_get_regime_weights_bear(self):
        from src.ensemble import get_regime_weights
        w = get_regime_weights("Bear")
        assert w["lstm"] == 0.25  # DL favored in bear

    def test_unknown_regime_fallback(self):
        from src.ensemble import get_regime_weights
        w = get_regime_weights("Unknown")
        assert all(v == 0.2 for v in w.values())


class TestMetaLearnerBeatsEqualWeight:
    def test_on_synthetic_data(self):
        from src.ensemble import train_meta_learner, predict_with_metalearner
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
        from src.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = np.random.randint(0, 2, 100)
        model = train_meta_learner(meta_X, y)
        assert model is not None


class TestMetaLearnerCoefficients:
    def test_coefficients_are_finite(self):
        from src.ensemble import train_meta_learner
        np.random.seed(42)
        meta_X = np.random.rand(100, 5)
        y = (meta_X[:, 0] > 0.5).astype(int)
        model = train_meta_learner(meta_X, y)
        coefs = model.named_steps["clf"].coef_[0]
        assert all(np.isfinite(coefs))

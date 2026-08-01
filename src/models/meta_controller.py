"""Meta-controller: combines 14 signal modules into one decision.

Uses regularized logistic regression with temporal cross-validation
to learn which combination of signals has historically predicted price moves.

The learned weights ARE the audit mechanism:
  high weight = module is currently predictive
  low weight = module is noise
  weight decay over time = signal is decaying

Usage:
    mc = MetaController()
    decision = mc.decide(signals)
    # {"action": "BUY", "position_size": 0.05, "confidence": 0.7, "reasoning": "..."}

    mc.train(ledger)
    # {"status": "trained", "accuracy": 0.58, "n_samples": 500}
"""

import logging
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit

from src.models.artifacts import (
    ArtifactBundle, ArtifactVerificationError, is_meta_controller_state,
)

logger = logging.getLogger(__name__)

SIGNAL_NAMES = [
    "ensemble_direction", "ensemble_confidence",
    "sentiment_score", "fii_net", "dii_net",
    "pcr", "mtf_signal",
    "regime_bull", "regime_bear",
    "var_95", "cvar_95", "sharpe",
    "volatility_forecast", "fundamental_score",
]

# Defaults — overridden by settings at runtime
MAX_POSITION_PCT = 0.10
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4
MIN_CONFIDENCE_TO_TRADE = 0.3
MIN_SAMPLES_TO_TRAIN = 500
META_CONTROLLER_C = 0.1


class MetaController:
    """Combines signal modules into one decision via regularized regression."""

    # Discount applied to confidence when model lacks calibration.
    # Uncalibrated logistic regression probabilities tend to be overconfident
    # near the decision boundary; this penalty compensates.
    UNCALIBRATED_CONFIDENCE_DISCOUNT = 0.7

    def __init__(self):
        self.model = None
        self.weights = None
        self.is_calibrated = False
        self._load_settings()

    def _load_settings(self):
        """Load configurable thresholds from settings."""
        try:
            from src.core.settings import settings
            self.max_position_pct = getattr(settings, "max_position_pct", MAX_POSITION_PCT)
            self.buy_threshold = getattr(settings, "buy_threshold", BUY_THRESHOLD)
            self.sell_threshold = getattr(settings, "sell_threshold", SELL_THRESHOLD)
            self.min_confidence_to_trade = getattr(settings, "min_confidence_to_trade", MIN_CONFIDENCE_TO_TRADE)
            self.min_samples_to_train = getattr(settings, "min_samples_to_train", MIN_SAMPLES_TO_TRAIN)
            self.meta_controller_c = getattr(settings, "meta_controller_c", META_CONTROLLER_C)
        except Exception:
            self.max_position_pct = MAX_POSITION_PCT
            self.buy_threshold = BUY_THRESHOLD
            self.sell_threshold = SELL_THRESHOLD
            self.min_confidence_to_trade = MIN_CONFIDENCE_TO_TRADE
            self.min_samples_to_train = MIN_SAMPLES_TO_TRAIN
            self.meta_controller_c = META_CONTROLLER_C

    @staticmethod
    def _can_calibrate(y: np.ndarray, min_cv: int = 3) -> int:
        """Return the maximum safe cv value for CalibratedClassifierCV.

        Calibration requires enough samples of each class in every fold.
        Returns 0 if calibration is not feasible, otherwise returns the
        largest cv <= min_cv that the class distribution supports.
        """
        if len(y) < min_cv * 2:
            return 0
        class_counts = np.bincount(y.astype(int))
        min_class_count = int(class_counts.min())
        if min_class_count < 2:
            return 0
        return min(min_cv, min_class_count)

    @staticmethod
    def _to_float(val, default=0.0) -> float:
        """Convert value to float, handling bytes from SQLite."""
        if val is None:
            return float(default)
        if isinstance(val, bytes):
            try:
                import struct
                val = struct.unpack("<d", val)[0]
            except (struct.error, ValueError):
                try:
                    val = float(int.from_bytes(val, byteorder="little"))
                except (ValueError, OverflowError):
                    return float(default)
        try:
            return float(val)
        except (TypeError, ValueError):
            return float(default)

    def extract_state_vector(self, signals: dict) -> np.ndarray:
        """Convert module outputs to fixed-size feature vector."""
        regime = signals.get("regime", "Sideways")
        regime_bull = 1.0 if regime == "Bull" else 0.0
        regime_bear = 1.0 if regime == "Bear" else 0.0

        vector = np.array([
            self._to_float(signals.get("ensemble_direction"), 0),
            self._to_float(signals.get("ensemble_confidence"), 0),
            self._to_float(signals.get("sentiment_score"), 0),
            self._to_float(signals.get("fii_net"), 0),
            self._to_float(signals.get("dii_net"), 0),
            self._to_float(signals.get("pcr"), 1.0),
            self._to_float(signals.get("mtf_signal"), 0),
            regime_bull,
            regime_bear,
            self._to_float(signals.get("var_95"), 0),
            self._to_float(signals.get("cvar_95"), 0),
            self._to_float(signals.get("sharpe"), 0),
            self._to_float(signals.get("volatility_forecast"), 0),
            self._to_float(signals.get("fundamental_score"), 0.5),
        ], dtype=np.float64)

        return vector

    def decide(self, signals: dict) -> dict:
        """Take all signal outputs, return one decision."""
        vector = self.extract_state_vector(signals)

        if self.model is None:
            return self._rule_based_decide(signals)

        proba = self.model.predict_proba(vector.reshape(1, -1))
        if proba.shape[1] < 2:
            return self._rule_based_decide(signals)
        prob = proba[0, 1]
        confidence = round(abs(prob - 0.5) * 2, 4)

        # Discount confidence when model lacks calibration to avoid
        # overconfident position sizing from raw logistic probabilities.
        if not self.is_calibrated:
            confidence = round(confidence * self.UNCALIBRATED_CONFIDENCE_DISCOUNT, 4)

        if confidence < self.min_confidence_to_trade:
            return {
                "action": "HOLD",
                "position_size": 0.0,
                "confidence": confidence,
                "reasoning": f"Confidence {confidence:.2f} below threshold {self.min_confidence_to_trade}",
            }

        if prob > self.buy_threshold:
            action = "BUY"
        elif prob < self.sell_threshold:
            action = "SELL"
        else:
            action = "HOLD"

        # Scale position size by confidence (higher confidence = larger position)
        if action != "HOLD":
            position_size = min(self.max_position_pct, confidence * self.max_position_pct)
        else:
            position_size = 0.0

        return {
            "action": action,
            "position_size": round(position_size, 4),
            "confidence": confidence,
            "reasoning": self.explain(signals),
        }

    def _rule_based_decide(self, signals: dict) -> dict:
        """Rule-based fallback when no model is trained yet."""
        ensemble_dir = signals.get("ensemble_direction")
        ensemble_conf = signals.get("ensemble_confidence", 0) or 0
        sentiment = signals.get("sentiment_score", 0) or 0
        regime = signals.get("regime", "Sideways")

        score = 0
        if ensemble_dir == 1:
            score += 0.4
        elif ensemble_dir == 0:
            score -= 0.4

        if sentiment > 0.2:
            score += 0.2
        elif sentiment < -0.2:
            score -= 0.2

        if regime == "Bull":
            score += 0.1
        elif regime == "Bear":
            score -= 0.1

        if score > 0.2:
            action = "BUY"
            position_size = min(self.max_position_pct, score * 0.2)
        elif score < -0.2:
            action = "SELL"
            position_size = min(self.max_position_pct, abs(score) * 0.2)
        else:
            action = "HOLD"
            position_size = 0.0

        return {
            "action": action,
            "position_size": round(position_size, 4),
            "confidence": round(min(1.0, abs(score)), 4),
            "reasoning": f"Rule-based: ensemble={ensemble_dir}, sentiment={sentiment:.2f}, regime={regime}",
        }

    def train(self, ledger) -> dict:
        """Retrain on ledger history using temporal cross-validation.

        Uses TimeSeriesSplit to prevent look-ahead bias and provide robust
        out-of-sample estimates. The final model is trained on all data
        with the regularization strength found by cross-validation.

        Calibration is attempted with an adaptive cv value. If the data
        is too small or imbalanced for reliable calibration, the model
        is stored as uncalibrated and confidence is discounted at
        decision time.
        """
        decisions = ledger.get_decisions()
        resolved = [d for d in decisions if d["actual_direction"] is not None]

        if len(resolved) < self.min_samples_to_train:
            return {
                "status": "insufficient_data",
                "n_samples": len(resolved),
                "required": self.min_samples_to_train,
            }

        # Reverse to chronological order (ledger returns DESC)
        resolved = list(reversed(resolved))

        X = np.array([self.extract_state_vector(d) for d in resolved])
        y = np.array([int(self._to_float(d["actual_direction"], 0)) for d in resolved])

        # Guard against single-class labels
        if len(np.unique(y)) < 2:
            return {
                "status": "single_class",
                "n_samples": len(resolved),
                "message": "All resolved decisions have the same direction",
            }

        # ── Temporal cross-validation ─────────────────────────────────────
        n_splits = min(5, max(2, len(X) // 100))
        tscv = TimeSeriesSplit(n_splits=n_splits)

        fold_accuracies = []
        fold_calibrated = []
        fold_models = []

        for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Skip folds where a class is missing
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                continue

            base = LogisticRegression(
                C=self.meta_controller_c,
                max_iter=1000,
                class_weight="balanced",
                solver="lbfgs",
            )
            base.fit(X_train, y_train)

            fold_cv = self._can_calibrate(y_train, min_cv=3)
            if fold_cv >= 2:
                try:
                    calibrated = CalibratedClassifierCV(base, cv=fold_cv)
                    calibrated.fit(X_train, y_train)
                    acc = calibrated.score(X_test, y_test)
                    fold_models.append(calibrated)
                    fold_calibrated.append(True)
                except Exception as e:
                    logger.warning("Fold %d calibration failed (cv=%d): %s — using uncalibrated",
                                   fold_idx, fold_cv, e)
                    acc = base.score(X_test, y_test)
                    fold_models.append(base)
                    fold_calibrated.append(False)
            else:
                acc = base.score(X_test, y_test)
                fold_models.append(base)
                fold_calibrated.append(False)
                logger.debug("Fold %d: skipped calibration (min_class_count too low)", fold_idx)

            fold_accuracies.append(acc)
            logger.debug("Fold %d: accuracy=%.3f calibrated=%s (train=%d, test=%d)",
                         fold_idx, acc, fold_calibrated[-1], len(train_idx), len(test_idx))

        if not fold_accuracies:
            return {
                "status": "cv_failed",
                "n_samples": len(X),
                "message": "Temporal cross-validation produced no valid folds",
            }

        mean_acc = np.mean(fold_accuracies)
        std_acc = np.std(fold_accuracies)
        n_calibrated = sum(fold_calibrated)

        # ── Train final model on all data ─────────────────────────────────
        final_model = LogisticRegression(
            C=self.meta_controller_c,
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
        )
        final_model.fit(X, y)

        final_cv = self._can_calibrate(y, min_cv=3)
        self.is_calibrated = False

        if final_cv >= 2:
            try:
                self.model = CalibratedClassifierCV(final_model, cv=final_cv)
                self.model.fit(X, y)
                self.weights = dict(zip(SIGNAL_NAMES, self.model.calibrated_classifiers_[0].estimator.coef_[0]))
                self.is_calibrated = True
            except Exception as e:
                logger.warning("Final calibration failed (cv=%d): %s — using uncalibrated model", final_cv, e)
                self.model = final_model
                self.weights = dict(zip(SIGNAL_NAMES, final_model.coef_[0]))
        else:
            logger.info("Skipping calibration: insufficient class samples (min_cv=%d)", final_cv)
            self.model = final_model
            self.weights = dict(zip(SIGNAL_NAMES, final_model.coef_[0]))

        logger.info(
            "Meta-controller trained: cv_accuracy=%.3f (+/- %.3f), n_samples=%d, "
            "folds=%d (calibrated=%d/%d), C=%.2f, model_calibrated=%s",
            mean_acc, std_acc, len(X), len(fold_accuracies), n_calibrated,
            len(fold_accuracies), self.meta_controller_c, self.is_calibrated,
        )

        return {
            "status": "trained",
            "accuracy": round(float(mean_acc), 4),
            "accuracy_std": round(float(std_acc), 4),
            "n_samples": len(X),
            "n_folds": len(fold_accuracies),
            "fold_accuracies": [round(float(a), 4) for a in fold_accuracies],
            "fold_calibrated": fold_calibrated,
            "is_calibrated": self.is_calibrated,
            "C": self.meta_controller_c,
        }

    def get_weights(self) -> dict:
        """Return learned weights sorted by absolute importance."""
        if self.weights is None:
            return {}
        return dict(sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True))

    def save(self, path: str = None, *, model_version: str = "1",
             feature_schema_version: str = "1",
             training_dataset_hash: str = "") -> str:
        """Save trained model to disk as a hash-verified artifact bundle.

        Returns the path of the written artifact.
        """
        if path is None:
            from src.core.constants import META_CONTROLLER_PATH
            path = META_CONTROLLER_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        import joblib
        joblib.dump({
            "model": self.model,
            "weights": self.weights,
            "is_calibrated": self.is_calibrated,
        }, path)
        bundle_name = Path(path).stem
        ArtifactBundle.create(
            str(Path(path).parent),
            bundle_name,
            {Path(path).name: ""},
            model_version=model_version,
            feature_schema_version=feature_schema_version,
            training_dataset_hash=training_dataset_hash,
        )
        logger.info("Meta-controller saved to %s (calibrated=%s)", path, self.is_calibrated)
        return path

    def load(self, path: str = None) -> bool:
        """Load trained model from disk. Returns True if loaded.

        The artifact is only deserialized after its SHA-256 digest matches
        the adjacent manifest. Legacy files without a manifest are refused.
        """
        if path is None:
            from src.core.constants import META_CONTROLLER_PATH
            path = META_CONTROLLER_PATH
        import os
        if not os.path.exists(path):
            return False
        try:
            bundle_name = Path(path).stem
            bundle = ArtifactBundle.load(str(Path(path).parent), bundle_name)
            state = bundle.load_joblib(Path(path).name, type_check=is_meta_controller_state)
            if not isinstance(state, dict) or "model" not in state or "weights" not in state:
                logger.warning("Invalid meta-controller state in %s — missing required keys", path)
                return False
            self.model = state.get("model")
            self.weights = state.get("weights")
            self.is_calibrated = state.get("is_calibrated", False)
            logger.info("Meta-controller loaded from %s (calibrated=%s)", path, self.is_calibrated)
            return True
        except ArtifactVerificationError as e:
            logger.warning("Refusing to load meta-controller %s: %s", path, e)
            return False
        except Exception as e:
            logger.warning("Failed to load meta-controller: %s", e)
            return False

    def explain(self, signals: dict) -> str:
        """Human-readable explanation of why this decision was made."""
        if self.weights is None:
            regime = signals.get("regime", "unknown")
            return f"Rule-based (no training data yet). Regime: {regime}"

        contributions = []
        for name, weight in self.weights.items():
            value = signals.get(name, 0) or 0
            if name == "regime_bull":
                value = 1.0 if signals.get("regime") == "Bull" else 0.0
            elif name == "regime_bear":
                value = 1.0 if signals.get("regime") == "Bear" else 0.0

            contribution = weight * value
            if abs(contribution) > 0.05:
                direction = "supports BUY" if contribution > 0 else "supports SELL"
                contributions.append(f"{name}: {direction} (w={weight:.3f})")

        return "; ".join(contributions) if contributions else "No strong signals"

"""Meta-controller: combines 14 signal modules into one decision.

Uses contextual bandit (stacked logistic regression) to learn which
combination of signals has historically predicted price moves.

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
import pickle
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

SIGNAL_NAMES = [
    "ensemble_direction", "ensemble_confidence",
    "sentiment_score", "fii_net", "dii_net",
    "pcr", "mtf_signal",
    "regime_bull", "regime_bear",
    "var_95", "cvar_95", "sharpe",
    "volatility_forecast", "fundamental_score",
]

MAX_POSITION_PCT = 0.10
BUY_THRESHOLD = 0.6
SELL_THRESHOLD = 0.4
MIN_CONFIDENCE_TO_TRADE = 0.3
MIN_SAMPLES_TO_TRAIN = 100


class MetaController:
    """Combines signal modules into one decision via contextual bandit."""

    def __init__(self):
        self.model = None
        self.weights = None

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

        prob = self.model.predict_proba(vector.reshape(1, -1))[0, 1]
        confidence = round(abs(prob - 0.5) * 2, 4)

        if confidence < MIN_CONFIDENCE_TO_TRADE:
            return {
                "action": "HOLD",
                "position_size": 0.0,
                "confidence": confidence,
                "reasoning": f"Confidence {confidence:.2f} below threshold {MIN_CONFIDENCE_TO_TRADE}",
            }

        if prob > BUY_THRESHOLD:
            action = "BUY"
        elif prob < SELL_THRESHOLD:
            action = "SELL"
        else:
            action = "HOLD"

        # Scale position size by confidence (higher confidence = larger position)
        if action != "HOLD":
            position_size = min(MAX_POSITION_PCT, confidence * MAX_POSITION_PCT)
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
            position_size = min(MAX_POSITION_PCT, score * 0.2)
        elif score < -0.2:
            action = "SELL"
            position_size = min(MAX_POSITION_PCT, abs(score) * 0.2)
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
        """Retrain on ledger history. Returns accuracy metrics."""
        decisions = ledger.get_decisions()
        resolved = [d for d in decisions if d["actual_direction"] is not None]

        if len(resolved) < MIN_SAMPLES_TO_TRAIN:
            return {
                "status": "insufficient_data",
                "n_samples": len(resolved),
                "required": MIN_SAMPLES_TO_TRAIN,
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

        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]

        self.model = LogisticRegression(C=1.0, max_iter=1000)
        self.model.fit(X_train, y_train)

        accuracy = self.model.score(X_test, y_test)
        self.weights = dict(zip(SIGNAL_NAMES, self.model.coef_[0]))

        logger.info(f"Meta-controller trained: accuracy={accuracy:.3f}, n_samples={len(X)}")
        return {
            "status": "trained",
            "accuracy": round(accuracy, 4),
            "n_samples": len(X),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def get_weights(self) -> dict:
        """Return learned weights sorted by absolute importance."""
        if self.weights is None:
            return {}
        return dict(sorted(self.weights.items(), key=lambda x: abs(x[1]), reverse=True))

    def save(self, path: str = None):
        """Save trained model to disk."""
        if path is None:
            from src.constants import META_CONTROLLER_PATH
            path = META_CONTROLLER_PATH
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "weights": self.weights}, f)
        logger.info("Meta-controller saved to %s", path)

    def load(self, path: str = None) -> bool:
        """Load trained model from disk. Returns True if loaded."""
        if path is None:
            from src.constants import META_CONTROLLER_PATH
            path = META_CONTROLLER_PATH
        import os
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as f:
                state = pickle.load(f)
            self.model = state.get("model")
            self.weights = state.get("weights")
            logger.info("Meta-controller loaded from %s", path)
            return True
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

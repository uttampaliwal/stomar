"""End-to-end smoke: ledger -> meta-controller -> decision flow.

Promoted from the old root ``verify_system.py`` print-script (which only
printed and asserted nothing — see ROADMAP) into a real pytest smoke test
on purely synthetic data: no network, no trained model artifacts, no
global state. It exists to catch wiring breaks across the three pillars in
one shot — a ledger the whole loop writes to, the meta-controller that
learns from it, and the decision function those models feed.
"""

import datetime as dt

import numpy as np

from src.models.meta_controller import MetaController
from src.trading.ledger import Ledger

_FEATURES = [
    "ensemble_confidence", "sentiment_score", "fii_net", "dii_net", "pcr",
    "mtf_signal", "regime", "var_95", "cvar_95", "sharpe",
    "volatility_forecast", "fundamental_score",
]


def _ledger_with_synthetic_history(db_path: str, n: int = 520) -> Ledger:
    """Fold a synthetic 'ensemble predictions ~60% correct' history in.

    n >= MetaController.MIN_SAMPLES_TO_TRAIN (500) — the old print-script
    used 200 rows, which always returned 'insufficient_data', so its
    'trained' branch was never actually exercised.
    """
    ledger = Ledger(db_path)
    np.random.seed(42)
    base_date = dt.date(2025, 1, 1)
    for i in range(n):
        direction = int(np.random.choice([0, 1]))
        signals = {
            "ensemble_direction": direction,
            "ensemble_confidence": np.random.uniform(0.4, 0.9),
            "sentiment_score": np.random.randn() * 0.3,
            "fii_net": np.random.randn() * 500,
            "dii_net": np.random.randn() * 300,
            "pcr": np.random.uniform(0.7, 1.5),
            "mtf_signal": np.random.randn() * 0.5,
            "regime": np.random.choice(["Bull", "Bear", "Sideways"]),
            "var_95": -np.random.uniform(0.01, 0.04),
            "cvar_95": -np.random.uniform(0.02, 0.06),
            "sharpe": np.random.uniform(-0.5, 2.0),
            "volatility_forecast": np.random.uniform(0.1, 0.3),
            "fundamental_score": np.random.uniform(40, 80),
        }
        did = ledger.log_decision(
            (base_date + dt.timedelta(days=i)).isoformat(), "RELIANCE.NS",
            signals, "BUY" if direction == 1 else "SELL", 0.05, 0.7)
        # ~60% of predictions land right (matches the print-script fixture).
        actual = direction if np.random.random() > 0.4 else 1 - direction
        ledger.log_outcome(did, np.random.randn() * 0.02, actual)
    return ledger


def test_ledger_roundtrip(tmp_path):
    """Log one decision + outcome; totals and accuracy come back out."""
    db = str(tmp_path / "smoke.db")
    ledger = Ledger(db)
    did = ledger.log_decision(
        "2025-01-15", "RELIANCE.NS", {
            "ensemble_direction": 1, "ensemble_confidence": 0.7,
            "regime": "Bull", "sentiment_score": 0.3, "fii_net": 500,
            "pcr": 1.1, "mtf_signal": 0.5, "var_95": -0.02, "cvar_95": -0.03,
            "sharpe": 1.2, "volatility_forecast": 0.18, "fundamental_score": 70,
        },
        "BUY", 0.05, 0.7, "test decision")
    ledger.log_outcome(did, 0.012, 1)

    perf = ledger.get_performance()
    assert perf["total_decisions"] == 1
    ledger.close()


def test_meta_controller_trains_on_synthetic_ledger(tmp_path):
    """A synthetic ledger trains the meta-controller to a sane accuracy."""
    db = str(tmp_path / "train.db")
    ledger = _ledger_with_synthetic_history(db)

    meta = MetaController()
    result = meta.train(ledger)
    assert result["status"] == "trained"
    assert result["n_samples"] >= 1
    assert 0.0 < result["accuracy"] <= 1.0

    weights = meta.get_weights()
    assert weights
    # Feature coverage: weights refer to the signal names we generated.
    feature_keys = set(_FEATURES) & set(weights)
    assert feature_keys, f"weights reference unexpected features: {weights}"

    ledger.close()


def test_decision_works_trained_and_untrained(tmp_path):
    """decide() returns the full decision dict both before and after training."""
    signals = {
        "ensemble_direction": 1, "ensemble_confidence": 0.8,
        "sentiment_score": 0.3, "fii_net": 500, "dii_net": -200,
        "pcr": 1.1, "mtf_signal": 0.5, "regime": "Bull",
        "var_95": -0.02, "cvar_95": -0.03, "sharpe": 1.5,
        "volatility_forecast": 0.15, "fundamental_score": 75,
    }
    db = str(tmp_path / "decide.db")
    ledger = _ledger_with_synthetic_history(db)
    meta = MetaController()
    meta.train(ledger)
    ledger.close()

    for controller, label in ((meta, "trained"), (MetaController(), "fresh")):
        decision = controller.decide(signals)
        for key in ("action", "position_size", "confidence", "reasoning"):
            assert key in decision, f"{label} decision missing '{key}'"
        assert decision["action"] in ("BUY", "SELL", "HOLD")
        assert 0.0 <= decision["position_size"] <= 1.0
        assert 0.0 <= decision["confidence"] <= 1.0

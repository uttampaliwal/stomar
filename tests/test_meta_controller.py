"""Tests for src/meta_controller.py."""

import os
import tempfile
from datetime import datetime, timedelta
import numpy as np
import pytest

from src.models.meta_controller import MetaController, SIGNAL_NAMES
from src.trading.ledger import Ledger


@pytest.fixture
def mc():
    return MetaController()


@pytest.fixture
def ledger_with_data():
    """Create a ledger with enough data to train."""
    db_path = os.path.join(tempfile.mkdtemp(), "test_mc.db")
    lg = Ledger(db_path)

    np.random.seed(42)
    for i in range(600):
        direction = int(i % 2)  # Alternate 0/1 to guarantee both classes
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
            "fundamental_score": np.random.uniform(0.4, 0.8),
        }
        decision_id = lg.log_decision(
            date=(datetime(2025, 1, 1) + timedelta(days=i)).strftime("%Y-%m-%d"),
            ticker="RELIANCE.NS",
            signals=signals,
            action="BUY" if direction == 1 else "SELL",
            position_size=0.05,
            confidence=0.7,
        )
        # Guarantee both classes in actual outcomes
        actual_dir = int(i % 2)
        lg.log_outcome(decision_id, actual_return=np.random.randn() * 0.02, actual_direction=actual_dir)

    yield lg
    lg.close()


# --- extract_state_vector ---

def test_extract_state_vector_shape(mc):
    signals = {
        "ensemble_direction": 1, "ensemble_confidence": 0.7,
        "sentiment_score": 0.3, "fii_net": 1000, "dii_net": -200,
        "pcr": 1.1, "mtf_signal": 0.5, "regime": "Bull",
        "var_95": -0.02, "cvar_95": -0.03, "sharpe": 1.0,
        "volatility_forecast": 0.18, "fundamental_score": 70,
    }
    vec = mc.extract_state_vector(signals)
    assert vec.shape == (14,)


def test_extract_state_vector_handles_missing(mc):
    vec = mc.extract_state_vector({})
    assert vec.shape == (14,)
    assert np.all(np.isfinite(vec))


def test_extract_state_vector_regime_encoding(mc):
    vec_bull = mc.extract_state_vector({"regime": "Bull"})
    vec_bear = mc.extract_state_vector({"regime": "Bear"})
    vec_side = mc.extract_state_vector({"regime": "Sideways"})
    assert vec_bull[7] == 1.0  # regime_bull
    assert vec_bull[8] == 0.0
    assert vec_bear[8] == 1.0  # regime_bear
    assert vec_bear[7] == 0.0
    assert vec_side[7] == 0.0
    assert vec_side[8] == 0.0


# --- decide without model ---

def test_decide_without_model_returns_action(mc):
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.7, "regime": "Bull"}
    decision = mc.decide(signals)
    assert "action" in decision
    assert decision["action"] in ("BUY", "SELL", "HOLD")
    assert "position_size" in decision
    assert "confidence" in decision
    assert "reasoning" in decision


def test_decide_without_model_strong_buy(mc):
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.8, "sentiment_score": 0.5, "regime": "Bull"}
    decision = mc.decide(signals)
    assert decision["action"] == "BUY"
    assert decision["position_size"] > 0


def test_decide_without_model_strong_sell(mc):
    signals = {"ensemble_direction": 0, "ensemble_confidence": 0.8, "sentiment_score": -0.5, "regime": "Bear"}
    decision = mc.decide(signals)
    assert decision["action"] == "SELL"
    assert decision["position_size"] > 0


def test_decide_without_model_neutral(mc):
    signals = {"ensemble_direction": None, "sentiment_score": 0.0, "regime": "Sideways"}
    decision = mc.decide(signals)
    assert decision["action"] == "HOLD"
    assert decision["position_size"] == 0.0


# --- decide with model ---

def test_decide_with_model(mc, ledger_with_data):
    mc.train(ledger_with_data)
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.7, "regime": "Bull",
               "sentiment_score": 0.3, "pcr": 1.1, "mtf_signal": 0.5}
    decision = mc.decide(signals)
    assert decision["action"] in ("BUY", "SELL", "HOLD")
    assert 0 <= decision["confidence"] <= 1


# --- position size capping ---

def test_position_size_capped(mc):
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.99, "sentiment_score": 1.0, "regime": "Bull"}
    decision = mc.decide(signals)
    assert decision["position_size"] <= 0.10


# --- train ---

def test_train_insufficient_data(mc):
    db_path = os.path.join(tempfile.mkdtemp(), "empty.db")
    lg = Ledger(db_path)
    result = mc.train(lg)
    lg.close()
    assert result["status"] == "insufficient_data"
    assert mc.model is None


def test_train_with_enough_data(mc, ledger_with_data):
    result = mc.train(ledger_with_data)
    assert result["status"] == "trained"
    assert result["accuracy"] > 0.0
    assert result["n_samples"] == 600
    assert mc.model is not None


# --- get_weights ---

def test_get_weights_empty(mc):
    assert mc.get_weights() == {}


def test_get_weights_returns_dict(mc, ledger_with_data):
    mc.train(ledger_with_data)
    weights = mc.get_weights()
    assert isinstance(weights, dict)
    assert len(weights) == len(SIGNAL_NAMES)
    for name in SIGNAL_NAMES:
        assert name in weights


def test_get_weights_sorted_by_importance(mc, ledger_with_data):
    mc.train(ledger_with_data)
    weights = mc.get_weights()
    keys = list(weights.keys())
    for i in range(len(keys) - 1):
        assert abs(weights[keys[i]]) >= abs(weights[keys[i + 1]])


# --- explain ---

def test_explain_without_model(mc):
    explanation = mc.explain({"regime": "Bull"})
    assert "Rule-based" in explanation


def test_explain_with_model(mc, ledger_with_data):
    mc.train(ledger_with_data)
    signals = {"ensemble_direction": 1, "ensemble_confidence": 0.7, "regime": "Bull",
               "sentiment_score": 0.3, "pcr": 1.1}
    explanation = mc.explain(signals)
    assert isinstance(explanation, str)
    assert len(explanation) > 0


# --- integration with ledger ---

def test_train_and_decide_flow(mc, ledger_with_data):
    result = mc.train(ledger_with_data)
    assert result["status"] == "trained"

    signals = {
        "ensemble_direction": 1, "ensemble_confidence": 0.75,
        "sentiment_score": 0.4, "fii_net": 800, "dii_net": -200,
        "pcr": 1.1, "mtf_signal": 0.6, "regime": "Bull",
        "var_95": -0.02, "cvar_95": -0.03, "sharpe": 1.5,
        "volatility_forecast": 0.15, "fundamental_score": 75,
    }
    decision = mc.decide(signals)
    assert decision["action"] in ("BUY", "SELL", "HOLD")
    assert len(decision["reasoning"]) > 0

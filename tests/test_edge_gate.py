"""Tests for scripts/edge_gate.py pure comparison math (no models)."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from scripts.edge_gate import (
    binomial_p_greater,
    economics,
    logistic_preds,
    mcnemar_p,
    momentum_preds,
    naive_preds,
    pooled_verdict,
    regime_breakdown,
    regime_labels,
    split_oos,
    strategy_rets,
)


def test_split_oos_contiguous_with_dead_zone():
    fit, evl = split_oos(100, 0.7, 0.3)
    assert fit == list(range(70))
    assert evl == list(range(70, 100))
    assert not set(fit) & set(evl)


def test_naive_preds_majority():
    assert naive_preds([1, 1, 0], 2) == [1, 1]
    assert naive_preds([0, 0, 1], 2) == [0, 0]


def test_momentum_preds_point_in_time():
    closes = [100, 101, 102, 103, 104, 105, 106]
    # i=6: close[5]=105 > close[0]=100 -> 1
    assert momentum_preds(closes, [6], lookback=5) == [1]
    closes2 = [106, 105, 104, 103, 102, 101, 100]
    assert momentum_preds(closes2, [6], lookback=5) == [0]


def test_logistic_preds_shape_and_values():
    rng = np.random.default_rng(0)
    X = rng.random((100, 6))
    y = (X[:, 0] > 0.5).astype(int)
    preds = logistic_preds(X[:70], y[:70], X[70:])
    assert len(preds) == 30
    assert set(preds) <= {0, 1}


def test_binomial_p_greater_extremes():
    assert binomial_p_greater(100, 100) < 1e-10
    assert binomial_p_greater(0, 100) > 0.99


def test_mcnemar_no_discordant_is_one():
    assert mcnemar_p([1, 0, 1], [1, 0, 1]) == 1.0


def test_mcnemar_strong_win_is_significant():
    ens = [1] * 20 + [0] * 5
    base = [0] * 20 + [0] * 5
    assert mcnemar_p(ens, base) < 0.05


def test_economics_buyhold_positive_on_uptrend():
    closes = list(100 * (1.001 ** np.arange(60)))
    econ = economics([1] * 60, closes)
    assert econ["total_return"] > 0
    assert econ["turnover"] == 1


def test_economics_flat_when_never_in_market():
    closes = list(100 * (1.001 ** np.arange(60)))
    econ = economics([0] * 60, closes)
    assert econ["total_return"] == 0.0
    assert econ["turnover"] == 0


def test_pooled_verdict_pass_and_fail():
    accs = {"ensemble": 0.55, "naive": 0.50, "momentum": 0.49, "logistic": 0.51}
    v = pooled_verdict(2000, 1100, accs, 0.10, 0.05, 0.01)
    assert v["passed"] is True
    assert all(v["checks"].values())

    bad = dict(accs, ensemble=0.49)
    v2 = pooled_verdict(2000, 980, bad, 0.10, 0.05, 0.6)
    assert v2["passed"] is False
    assert v2["checks"]["beats_naive"] is False


def test_strategy_rets_length_and_flat():
    closes = [100.0, 101.0, 102.0, 101.5]
    r = strategy_rets([0, 0, 0, 0], closes)
    assert len(r) == 4
    assert float(np.sum(np.abs(r))) == 0.0


def test_regime_labels_band_and_unknown():
    import pandas as pd

    dates = pd.bdate_range("2024-01-01", periods=100)
    assert regime_labels([], lookback=63) == []
    labels = regime_labels([str(d.date()) for d in dates[:5]], lookback=63)
    assert labels == ["unknown"] * 5  # not enough history


def test_regime_breakdown_pools_correctly(monkeypatch):
    import scripts.edge_gate as eg

    dates = [f"2024-01-{d:02d}" for d in range(1, 11)]
    det = {"dates": dates,
           "actual": [1] * 10,
           "preds": {"ensemble": [1] * 10, "naive": [1] * 10,
                     "momentum": [0] * 10, "logistic": [1] * 10},
           "closes": [100.0 + i for i in range(10)]}
    per_ticker = [{"eval_detail": det}]
    monkeypatch.setattr(eg, "regime_labels",
                        lambda dates, lookback=63, band=0.03: ["bull"] * len(dates))
    out = regime_breakdown(per_ticker)
    assert set(out) == {"bull"}
    assert out["bull"]["n"] == 10
    assert out["bull"]["accs"]["ensemble"] == 1.0
    assert out["bull"]["accs"]["momentum"] == 0.0

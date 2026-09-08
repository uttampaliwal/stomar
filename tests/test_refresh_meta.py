"""Fast tests for F1 refresh guard logic (no bundles needed)."""

import numpy as np

from scripts.refresh_meta import _equal_weight_preds, _meta_X_y, decide_promotion


def test_decide_promotion_strict():
    assert decide_promotion(0.60, 0.59) is True
    assert decide_promotion(0.59, 0.59) is False
    assert decide_promotion(0.50, 0.59) is False


def test_equal_weight_preds_majority_vote():
    X = np.array([[0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                  [0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                  [0.9, 0.1, 0.1, 0.1, 0.1, 0.1]])
    assert list(_equal_weight_preds(X)) == [1, 0, 0]


def test_meta_X_y_shapes_and_order():
    rows = [{"xgb_prob": 0.6, "lgb_prob": 0.5, "lstm_prob": 0.4,
             "gru_prob": 0.3, "transformer_prob": 0.2, "cat_prob": 0.1,
             "actual": 1}]
    X, y = _meta_X_y(rows)
    assert X.shape == (1, 6)
    assert list(X[0]) == [0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    assert list(y) == [1]

"""Tests for src/models/calibration.py (P4.4)."""


from src.models.calibration import reliability_report


def _decisions(n=100, seed=42):
    import random
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        conf = rng.uniform(0.05, 0.95)
        correct = rng.random() < conf  # perfectly calibrated by construction
        rows.append({
            "id": i,
            "confidence": round(conf, 4),
            "correct": 1 if correct else 0,
            "actual_return": 0.01 if correct else -0.01,
            "action": "BUY" if i % 2 == 0 else "SELL",
        })
    return rows


def test_perfectly_calibrated_data_has_low_ece():
    report = reliability_report(_decisions(500, seed=1))
    assert report["n"] == 500
    assert report["ece"] is not None
    assert report["ece"] < 0.05, report["ece"]  # near-perfect calibration
    assert len(report["bins"]) == 5
    assert all(b["n"] > 0 for b in report["bins"])


def test_empty_input():
    report = reliability_report([])
    assert report["n"] == 0
    assert report["ece"] is None
    assert "note" in report


def test_unresolved_decisions_excluded():
    rows = _decisions(10)
    rows[0]["actual_return"] = None  # unresolved — must be excluded
    rows[1]["correct"] = None
    report = reliability_report(rows)
    assert report["n"] == 8


def test_bin_counts_sum_to_n():
    report = reliability_report(_decisions(123, seed=7))
    assert sum(b["n"] for b in report["bins"]) == 123

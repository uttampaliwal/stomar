"""Confidence calibration checks (P4.4).

A confidence of 68% is only trustworthy if the system is right roughly
68% of the time at that confidence. This module computes reliability
diagrams (calibration curves) from resolved ledger decisions.

Usage:
    from src.models.calibration import reliability_report
    report = reliability_report(decisions)
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)

# Confidence bins: 0-20%, 20-40%, 40-60%, 60-80%, 80-100%
DEFAULT_BINS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]


def _bin_index(conf: float, bins: list[tuple]) -> int:
    for i, (lo, hi) in enumerate(bins):
        if lo <= conf < hi:
            return i
    return len(bins) - 1  # conf == 1.0 falls in the last bin


def reliability_report(decisions: list[dict],
                       bins: list[tuple] = None) -> dict:
    """Compute a reliability diagram from resolved decisions.

    Only decisions with a logged outcome (``actual_return`` / ``correct``)
    are used. Bins with no decisions are reported as empty.

    Args:
        decisions: Ledger decisions (live only — backfill is in-sample).
        bins: Confidence bin edges, default 5 bins of 0.2 width.

    Returns:
        Dict with per-bin accuracy/confidence/count, ECE, MCE, and n.
    """
    bins = bins or DEFAULT_BINS
    resolved = [
        d for d in decisions
        if d.get("actual_return") is not None
        and d.get("confidence") is not None
        and d.get("correct") is not None
    ]
    if not resolved:
        return {
            "n": 0,
            "bins": [],
            "ece": None,
            "mce": None,
            "note": "no resolved decisions available for calibration",
        }

    confs = np.array([float(d["confidence"]) for d in resolved], dtype=np.float64)
    corrects = np.array([1 if d["correct"] else 0 for d in resolved], dtype=np.float64)

    bin_rows = []
    for i, (lo, hi) in enumerate(bins):
        mask = (confs >= lo) & (confs < hi)
        n = int(mask.sum())
        if n == 0:
            bin_rows.append({
                "bin": f"{lo:.0%}-{hi:.0%}",
                "n": 0,
                "mean_confidence": None,
                "empirical_accuracy": None,
            })
            continue
        mean_conf = float(confs[mask].mean())
        emp_acc = float(corrects[mask].mean())
        bin_rows.append({
            "bin": f"{lo:.0%}-{hi:.0%}",
            "n": n,
            "mean_confidence": round(mean_conf, 4),
            "empirical_accuracy": round(emp_acc, 4),
        })

    # Expected Calibration Error: weighted mean |empirical - confidence|
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.array([r["n"] for r in bin_rows if r["n"] > 0], dtype=np.float64)
        diffs = np.array([
            abs(r["empirical_accuracy"] - r["mean_confidence"])
            for r in bin_rows if r["n"] > 0
        ], dtype=np.float64)
        if len(weights) > 0 and weights.sum() > 0:
            ece = float((weights * diffs).sum() / weights.sum())
            mce = float(diffs.max()) if len(diffs) > 0 else None
        else:
            ece = None
            mce = None

    overall_acc = float(corrects.mean())
    return {
        "n": len(resolved),
        "overall_accuracy": round(overall_acc, 4),
        "mean_confidence": round(float(confs.mean()), 4),
        "bins": bin_rows,
        "ece": round(ece, 4) if ece is not None else None,
        "mce": round(mce, 4) if mce is not None else None,
        "note": ("ECE is the average gap between stated confidence and actual "
                 "accuracy; closer to 0 is better. Requires resolved live data."),
    }

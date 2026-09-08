"""F1 signal loop: refit stale per-ticker meta-learners on recent windows.

The O1b gate showed a fresh logistic on the same six base probs beats the
deployed per-ticker meta-learners on 19/20 tickers: the combiner is stale,
not the base models. This script refits each ticker's meta-learner on its
most recent backtest rows (same rows the gate scores, via
``edge_gate.load_ticker_rows``) with a holdout guard:

  fit on first 70% of the recent window, promote the refit ONLY if it beats
  the deployed meta on the last-30% holdout (or beats equal-weight when no
  deployed meta exists). Otherwise the old artifact stays.

Threshold sweep on the holdout is reported as a diagnostic only (F2/O5
input) — the gate verdict stays on the frozen 0.5 rule for comparability.

Usage:
    uv run scripts/refresh_meta.py
    uv run scripts/refresh_meta.py --tickers RELIANCE.NS INFY.NS --window 750
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.logging_config import get_logger  # noqa: E402

logger = get_logger("refresh_meta")

WINDOW_DEFAULT = 750
FIT_FRAC = 0.7


def _meta_X_y(rows):
    X = np.array([[r["xgb_prob"], r["lgb_prob"], r["lstm_prob"],
                   r["gru_prob"], r["transformer_prob"], r["cat_prob"]]
                  for r in rows], dtype=float)
    y = np.array([r["actual"] for r in rows], dtype=int)
    return X, y


def _equal_weight_preds(X):
    return (X.mean(axis=1) > 0.5).astype(int)


def decide_promotion(new_acc: float, baseline_acc: float) -> bool:
    """Promote the refit only on strict holdout improvement."""
    return bool(new_acc > baseline_acc)


def refresh_ticker(ticker: str, window: int = WINDOW_DEFAULT) -> dict | None:
    from scripts.edge_gate import load_ticker_rows
    from src.models.ensemble import (
        load_meta_model,
        predict_with_metalearner,
        save_meta_model,
        train_meta_learner,
    )
    from src.core.constants import MODELS_DIR

    rows, closes, _ = load_ticker_rows(ticker)
    if not rows:
        return None
    rows = rows[-window:]
    X, y = _meta_X_y(rows)
    n = len(X)
    cut = int(n * FIT_FRAC)
    X_fit, y_fit = X[:cut], y[:cut]
    X_hold, y_hold = X[cut:], y[cut:]
    if len(X_hold) < 30 or len(np.unique(y_fit)) < 2:
        logger.warning("skip %s: holdout too small or single-class fit", ticker)
        return None

    new_meta = train_meta_learner(X_fit, y_fit)
    new_pred = (predict_with_metalearner(new_meta, X_hold) > 0.5).astype(int)
    new_acc = float((new_pred == y_hold).mean())

    clean = ticker.replace(".", "_")
    meta_path = os.path.join(MODELS_DIR, f"meta_{clean}.pkl")
    old_acc, old_present = None, False
    try:
        old_meta = load_meta_model(meta_path)
        old_pred = (predict_with_metalearner(old_meta, X_hold) > 0.5).astype(int)
        old_acc = float((old_pred == y_hold).mean())
        old_present = True
    except Exception as exc:
        logger.debug("no loadable deployed meta for %s: %s", ticker, exc)

    ew_acc = float((_equal_weight_preds(X_hold) == y_hold).mean())
    baseline_acc = old_acc if old_present else ew_acc
    baseline_name = "deployed-meta" if old_present else "equal-weight"
    promote = decide_promotion(new_acc, baseline_acc)

    # Threshold diagnostic on holdout (report only).
    sweep = {}
    try:
        from src.trading.simulate import sweep_entry_thresholds

        hold_closes = closes[-len(X_hold):]
        new_probs = predict_with_metalearner(new_meta, X_hold)
        import pandas as pd

        tbl = sweep_entry_thresholds(pd.Series(hold_closes), pd.Series(new_probs))
        best = tbl.loc[tbl["total_return"].idxmax()]
        sweep = {"best_threshold": float(best.name),
                 "total_return": float(best["total_return"]),
                 "sharpe": float(best["sharpe"]),
                 "turnover": int(best["turnover"])}
    except Exception as exc:
        logger.debug("threshold sweep failed for %s: %s", ticker, exc)

    if promote:
        first = rows[0].get("date", "")
        last = rows[-1].get("date", "")
        save_meta_model(new_meta, meta_path, feature_schema_version="4",
                        training_dataset_hash=f"{first}:{last}:{n}")

    return {"ticker": ticker, "n_window": n, "n_holdout": len(X_hold),
            "new_acc": round(new_acc, 4), "baseline": baseline_name,
            "baseline_acc": round(float(baseline_acc), 4),
            "equal_weight_acc": round(ew_acc, 4),
            "promoted": promote, "threshold_sweep": sweep}


def main() -> int:
    ap = argparse.ArgumentParser(description="F1: refit stale meta-learners")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--window", type=int, default=WINDOW_DEFAULT)
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "data", "edge_gate"))
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        from src.data.data_fetcher import NSE_STOCKS

        tickers = [t for t in NSE_STOCKS]

    results = []
    for t in tickers:
        logger.info("refresh-meta: %s", t)
        try:
            res = refresh_ticker(t, args.window)
        except Exception as exc:
            logger.warning("refresh-meta: %s failed (%s)", t, exc)
            continue
        if res:
            results.append(res)
            print(f"{t:15s} new={res['new_acc']:.3f} {res['baseline']}={res['baseline_acc']:.3f} "
                  f"promoted={'YES' if res['promoted'] else 'no '}"
                  + (f" thr={res['threshold_sweep'].get('best_threshold')}" if res["threshold_sweep"] else ""))

    if not results:
        print("REFRESH: NO DATA")
        return 2

    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.report_dir, f"meta_refresh_{stamp}.json"), "w") as f:
        json.dump({"stamp": stamp, "window": args.window, "tickers": results},
                  f, indent=2, default=str)

    n_prom = sum(1 for r in results if r["promoted"])
    mean_new = float(np.mean([r["new_acc"] for r in results]))
    mean_base = float(np.mean([r["baseline_acc"] for r in results]))
    print(f"\nREFRESH: promoted {n_prom}/{len(results)} "
          f"(mean holdout new={mean_new:.3f} vs baseline={mean_base:.3f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

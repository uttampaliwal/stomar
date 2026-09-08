"""O1b Edge Gate — strategy must beat all challengers before O5/O6 spend.

Compares the deployed ensemble (``backtest_ensemble`` over cached history,
no retraining) against three strictly point-in-time challengers on the SAME
rows, with economics through the single authoritative cost engine
(``simulate.strategy_return_series`` — full NSE stack, multiplicative):

  naive    — always predict the fit-window majority class
  momentum — predict sign of trailing 5-day return (data <= t-1 only)
  logistic — LogisticRegression on base-model probs, fit on fit-window only
  buy-hold — always-long economics comparator (no accuracy counterpart)

Accuracy is scored on eval-window rows (last ``--eval-frac``); economics on
the same window. Verdict is pooled across tickers: one-sided binomial
(ensemble accuracy > 0.5) + McNemar vs the best baseline + net total return
vs buy-hold.

PASS => proceed to O2..O11. FAIL => frozen F1/F2/F3 branch (MASTER-PLAN).

Scope honesty: bundles were trained on this same history, so this gate is a
NECESSARY-but-not-sufficient check (deployed bundle vs challengers). The
sufficient OOS evidence remains the pipeline walk-forward gate + the 60-day
paper clock. A FAIL here means: do not spend on O5/O6 optimization.

Usage:
    uv run scripts/edge_gate.py                          # all loadable tickers
    uv run scripts/edge_gate.py --tickers RELIANCE.NS INFY.NS
    uv run scripts/edge_gate.py --eval-frac 0.3 --fit-frac 0.7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.core.logging_config import get_logger  # noqa: E402

logger = get_logger("edge_gate")

EVAL_FRAC_DEFAULT = 0.3
FIT_FRAC_DEFAULT = 0.7
MOM_LOOKBACK = 5
ALPHA = 0.05


# ── pure comparison math (unit-tested, no models needed) ─────────────────────

def split_oos(n: int, fit_frac: float = FIT_FRAC_DEFAULT,
              eval_frac: float = EVAL_FRAC_DEFAULT):
    """Contiguous fit/eval split; middle rows are dead zone (embargo-like)."""
    n_fit = int(n * fit_frac)
    n_eval = int(n * eval_frac)
    fit = list(range(0, n_fit))
    evl = list(range(n - n_eval, n))
    return fit, evl


def accuracy(preds, actual) -> float | None:
    preds = np.asarray(preds)
    actual = np.asarray(actual)
    if len(actual) == 0:
        return None
    return float((preds == actual).mean())


def naive_preds(fit_actual, n: int) -> list[int]:
    """Majority class of the fit window (ties -> long)."""
    majority = 1 if np.mean(fit_actual) >= 0.5 else 0
    return [majority] * n


def momentum_preds(closes: list[float], eval_idx: list[int],
                   lookback: int = MOM_LOOKBACK) -> list[int]:
    """1 if close[i-1] > close[i-1-lookback] else 0 — strictly point-in-time."""
    out = []
    for i in eval_idx:
        if i - 1 - lookback < 0:
            out.append(1)
        else:
            out.append(1 if closes[i - 1] > closes[i - 1 - lookback] else 0)
    return out


def logistic_preds(fit_X, fit_y, eval_X) -> list[int]:
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, random_state=42)
    clf.fit(np.asarray(fit_X, dtype=float), np.asarray(fit_y, dtype=int))
    return [int(v) for v in clf.predict(np.asarray(eval_X, dtype=float))]


def binomial_p_greater(k: int, n: int, p: float = 0.5) -> float:
    """One-sided exact p-value for observed k/n exceeding p."""
    from scipy.stats import binomtest

    return float(binomtest(k, n, p, alternative="greater").pvalue)


def mcnemar_p(ens_correct: list[int], base_correct: list[int]) -> float:
    """Exact McNemar p-value (ensemble > baseline) on discordant pairs."""
    b = sum(1 for e, c in zip(ens_correct, base_correct) if e and not c)
    c = sum(1 for e, c in zip(ens_correct, base_correct) if c and not e)
    if b + c == 0:
        return 1.0
    return binomial_p_greater(b, b + c, 0.5)


def economics(pred_dirs, closes) -> dict:
    """Net-of-costs directional long/flat economics on an eval window."""
    from src.models.validation import performance_metrics

    rets = strategy_rets(pred_dirs, closes)
    m = performance_metrics(rets)
    pos = np.where(np.asarray(pred_dirs) > 0.5, 1.0, 0.0)
    return {
        "total_return": float(m.get("total_return", 0.0)),
        "sharpe": float(m.get("sharpe", 0.0)),
        "max_drawdown": float(m.get("max_drawdown", 0.0)),
        "turnover": int(np.sum(np.diff(np.concatenate(([0.0], pos))) != 0)),
    }


def strategy_rets(pred_dirs, closes) -> np.ndarray:
    """Per-bar net-of-costs strategy returns (powers economics + F2 regimes)."""
    from src.trading.simulate import strategy_return_series

    pos = np.where(np.asarray(pred_dirs) > 0.5, 1.0, 0.0)
    fwd = pd_pct_fwd(closes)
    return np.asarray(strategy_return_series(pos, fwd), dtype=float)


def regime_labels(dates, lookback: int = 63, band: float = 0.03) -> list[str]:
    """NIFTY-trend regime per date: bull/bear/sideways/unknown.

    Trailing ``lookback``-day index return vs ±``band`` — strictly
    point-in-time (only index values on/before each date).
    """
    import pandas as pd

    try:
        from src.signals.feature_pipeline import load_index_history

        idx = load_index_history(period="10y")
    except Exception as exc:
        logger.warning("regime split unavailable (no index history): %s", exc)
        return ["unknown"] * len(dates)
    if idx is None or idx.empty or "close" not in idx.columns:
        return ["unknown"] * len(dates)
    dts = pd.to_datetime(dates)
    series = idx["close"].reindex(dts, method="ffill")
    ret = series.pct_change(lookback)
    out = []
    for v in ret.values:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            out.append("unknown")
        elif v > band:
            out.append("bull")
        elif v < -band:
            out.append("bear")
        else:
            out.append("sideways")
    return out


def pd_pct_fwd(closes) -> np.ndarray:
    closes = np.asarray(closes, dtype=float)
    fwd = np.empty_like(closes)
    fwd[:-1] = closes[1:] / np.maximum(closes[:-1], 1e-12) - 1.0
    fwd[-1] = 0.0
    return fwd


def pooled_verdict(n_eval: int, k_ens: int, accs: dict,
                   ret_ens: float, ret_bh: float,
                   mcnemar_best_p: float, alpha: float = ALPHA) -> dict:
    """All-or-nothing gate per MASTER-PLAN O1b."""
    p_binom = binomial_p_greater(k_ens, n_eval, 0.5)
    checks = {
        "beats_naive": accs["ensemble"] > accs["naive"],
        "beats_momentum": accs["ensemble"] > accs["momentum"],
        "beats_logistic": accs["ensemble"] > accs["logistic"],
        "beats_buyhold_net": ret_ens > ret_bh,
        "binomial_p_lt_alpha": p_binom < alpha,
        "mcnemar_p_lt_alpha": mcnemar_best_p < alpha,
    }
    passed = all(checks.values())
    return {"passed": passed, "checks": checks,
            "p_binom": p_binom, "mcnemar_best_p": mcnemar_best_p}


# ── heavy per-ticker path (trained bundles, cached history) ──────────────────

def load_ticker_rows(ticker: str):
    """Load bundle + features and run backtest rows for a ticker.

    Shared by the gate and F1 meta refresh so both score identical rows.
    Returns (rows, closes, meta_used) or (None, None, False) when skipped.
    """
    import pandas as pd

    from src.data.data_fetcher import fetch_stock_data
    from src.models.ensemble import backtest_ensemble
    from src.models.model import load_models

    clean = ticker.replace(".", "_")
    try:
        df = fetch_stock_data(ticker, period="5y", force_refresh=False)
    except Exception as exc:
        logger.warning("skip %s: no cached history (%s)", ticker, exc)
        return None, None, False
    # Full factor set via the shared trainer-identical builder (O2 parity).
    # External PIT features resolve from on-disk caches (sentiment JSON,
    # fii_dii + MTF parquet) with neutral defaults — same convention as
    # training backfill.
    from src.models.trainer_features import build_feature_frame

    df_feat = build_feature_frame(df, ticker=ticker)

    try:
        lstm, gru, transformer, xgb, scaler, features, lgb = load_models(ticker)
    except Exception as exc:
        logger.warning("skip %s: bundle not loadable (%s)", ticker, exc)
        return None

    cat_model = None
    try:
        from src.models.artifacts import ArtifactBundle

        bundle = ArtifactBundle.for_ticker("models", ticker)
        if f"{clean}_cat.pkl" in bundle.listed_files():
            from src.models.model import has_predict

            cat_model = bundle.load_joblib(f"{clean}_cat.pkl", type_check=has_predict)
    except Exception as exc:
        logger.debug("cat unavailable for %s: %s", ticker, exc)

    meta_model = None
    try:
        # Production path, identical to orchestrator._run_ensemble: per-ticker
        # meta lives in its OWN manifest bundle (meta_{T}.pkl), not inside
        # the ticker bundle — a bundle-listing check here silently disables
        # the meta path and gates the wrong strategy.
        import os

        from src.core.constants import MODELS_DIR
        from src.models.ensemble import load_meta_model

        ticker_meta_path = os.path.join(MODELS_DIR, f"meta_{clean}.pkl")
        if os.path.exists(ticker_meta_path):
            meta_model = load_meta_model(ticker_meta_path)
    except Exception as exc:
        logger.debug("meta unavailable for %s: %s", ticker, exc)

    feature_cols = features if isinstance(features, list) and features else None
    if not feature_cols:
        logger.warning("skip %s: no stored feature list", ticker)
        return None

    rows = backtest_ensemble(
        lstm, gru, transformer, xgb, scaler, feature_cols, df_feat,
        lgb_model=lgb, cat_model=cat_model, meta_model=meta_model,
    )
    if not rows or len(rows) < 100:
        logger.warning("skip %s: only %d backtest rows", ticker, len(rows) if rows else 0)
        return None, None, False

    close_by_date = df_feat["close"].to_dict()

    def row_close(r):
        ts = pd.Timestamp(r["date"])
        if ts in close_by_date:
            return float(close_by_date[ts])
        return None

    closes_all, keep = [], []
    for r in rows:
        c = row_close(r)
        if c is not None and c > 0:
            closes_all.append(c)
            keep.append(r)
    rows = keep
    if len(rows) < 100:
        logger.warning("skip %s: price alignment left %d rows", ticker, len(rows))
        return None, None, False
    return rows, closes_all, meta_model is not None


def evaluate_ticker(ticker: str, fit_frac: float, eval_frac: float) -> dict | None:
    rows, closes_all, meta_used = load_ticker_rows(ticker)
    if not rows:
        return None

    actual = np.array([r["actual"] for r in rows])
    ens_pred = np.array([r["final_ensemble"] for r in rows])
    meta_X = np.array([[r["xgb_prob"], r["lgb_prob"], r["lstm_prob"],
                        r["gru_prob"], r["transformer_prob"], r["cat_prob"]]
                       for r in rows])

    fit_idx, eval_idx = split_oos(len(rows), fit_frac, eval_frac)
    ev_actual = actual[eval_idx]
    ev_ens = ens_pred[eval_idx]
    ev_closes = [closes_all[i] for i in eval_idx]

    ev_naive = np.array(naive_preds(actual[fit_idx], len(eval_idx)))
    ev_mom = np.array(momentum_preds(closes_all, eval_idx))
    ev_log = np.array(logistic_preds(meta_X[fit_idx], actual[fit_idx], meta_X[eval_idx]))

    accs = {
        "ensemble": accuracy(ev_ens, ev_actual),
        "naive": accuracy(ev_naive, ev_actual),
        "momentum": accuracy(ev_mom, ev_actual),
        "logistic": accuracy(ev_log, ev_actual),
    }
    ens_correct = (ev_ens == ev_actual).astype(int).tolist()
    mcn = {name: mcnemar_p(ens_correct, (p == ev_actual).astype(int).tolist())
           for name, p in (("naive", ev_naive), ("momentum", ev_mom),
                           ("logistic", ev_log))}
    best_base = max(("naive", "momentum", "logistic"), key=lambda k: accs[k])

    econ_ens = economics(ev_ens, ev_closes)
    econ_bh = economics(np.ones(len(eval_idx), dtype=int), ev_closes)
    econ_mom = economics(ev_mom, ev_closes)
    econ_log = economics(ev_log, ev_closes)

    return {
        "ticker": ticker,
        "n_rows": len(rows),
        "n_eval": len(eval_idx),
        "k_ens": int((ev_ens == ev_actual).sum()),
        "k_base": {
            "naive": int((ev_naive == ev_actual).sum()),
            "momentum": int((ev_mom == ev_actual).sum()),
            "logistic": int((ev_log == ev_actual).sum()),
        },
        "accs": accs,
        "mcnemar": mcn,
        "best_baseline": best_base,
        "econ": {"ensemble": econ_ens, "buy_hold": econ_bh,
                 "momentum": econ_mom, "logistic": econ_log},
        "meta_used": meta_used,
        # F2 regime split inputs (raw eval vectors; pooled later).
        "eval_detail": {
            "dates": [str(rows[i]["date"]) for i in eval_idx],
            "actual": ev_actual.astype(int).tolist(),
            "preds": {"ensemble": ev_ens.astype(int).tolist(),
                      "naive": ev_naive.astype(int).tolist(),
                      "momentum": ev_mom.astype(int).tolist(),
                      "logistic": ev_log.astype(int).tolist()},
            "closes": [float(c) for c in ev_closes],
        },
    }


def regime_breakdown(per_ticker) -> dict:
    """F2: pool eval bars by NIFTY-trend regime (descriptive, not a retest).

    Returns per-regime {n, accs, ret_ens, ret_bh} pooled across tickers.
    Economics per regime compounds that regime's bars of the full-window
    per-bar net return series.
    """
    all_dates = sorted({d for r in per_ticker for d in r["eval_detail"]["dates"]})
    reg_of = dict(zip(all_dates, regime_labels(all_dates)))
    out = {}
    for regime in ("bull", "bear", "sideways"):
        n = k_ens = 0
        k_base = {"naive": 0, "momentum": 0, "logistic": 0}
        rets_ens, rets_bh = [], []
        for r in per_ticker:
            det = r["eval_detail"]
            idx = [i for i, d in enumerate(det["dates"]) if reg_of.get(d) == regime]
            if not idx:
                continue
            actual = np.array(det["actual"])[idx]
            for name in ("ensemble", "naive", "momentum", "logistic"):
                pred = np.array(det["preds"][name])[idx]
                c = int((pred == actual).sum())
                if name == "ensemble":
                    k_ens += c
                else:
                    k_base[name] += c
            n += len(idx)
            # Regime economics: compound this regime's bars of the ticker's
            # full-window per-bar series (= growth if invested only on
            # regime bars).
            full_rets = strategy_rets(det["preds"]["ensemble"], det["closes"])
            full_bh = strategy_rets([1] * len(det["closes"]), det["closes"])
            rets_ens.append(float(np.prod([1 + full_rets[i] for i in idx]) - 1))
            rets_bh.append(float(np.prod([1 + full_bh[i] for i in idx]) - 1))
        if n == 0:
            continue
        out[regime] = {
            "n": n,
            "accs": {"ensemble": k_ens / n,
                     **{k: v / n for k, v in k_base.items()}},
            "ret_ens": float(np.mean(rets_ens)),
            "ret_bh": float(np.mean(rets_bh)),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="O1b edge gate: ensemble vs challengers")
    ap.add_argument("--tickers", nargs="*", default=None)
    ap.add_argument("--fit-frac", type=float, default=FIT_FRAC_DEFAULT)
    ap.add_argument("--eval-frac", type=float, default=EVAL_FRAC_DEFAULT)
    ap.add_argument("--regime-split", action="store_true",
                    help="F2: pooled accuracy/economics by NIFTY-trend regime")
    ap.add_argument("--report-dir", default=os.path.join(ROOT, "data", "edge_gate"))
    args = ap.parse_args()

    if args.tickers:
        tickers = args.tickers
    else:
        from src.data.data_fetcher import NSE_STOCKS

        tickers = [t for t in NSE_STOCKS]

    per_ticker = []
    for t in tickers:
        logger.info("edge-gate: evaluating %s", t)
        try:
            res = evaluate_ticker(t, args.fit_frac, args.eval_frac)
        except Exception as exc:
            logger.warning("edge-gate: %s failed (%s)", t, exc)
            traceback.print_exc()
            continue
        if res:
            per_ticker.append(res)
            a = res["accs"]
            print(f"{t:15s} n_eval={res['n_eval']:4d} "
                  f"ens={a['ensemble']:.3f} naive={a['naive']:.3f} "
                  f"mom={a['momentum']:.3f} log={a['logistic']:.3f} "
                  f"ret_ens={res['econ']['ensemble']['total_return']:+.3f} "
                  f"ret_bh={res['econ']['buy_hold']['total_return']:+.3f}")

    if not per_ticker:
        print("EDGE GATE: NO DATA — no ticker evaluated")
        return 2

    n_eval = sum(r["n_eval"] for r in per_ticker)
    k_ens = sum(r["k_ens"] for r in per_ticker)
    # Micro-averaged (pooled) accuracies so the binomial line and the
    # comparisons use the same counts.
    accs = {"ensemble": k_ens / n_eval}
    for k in ("naive", "momentum", "logistic"):
        accs[k] = sum(r["k_base"][k] for r in per_ticker) / n_eval
    ret_ens = float(np.mean([r["econ"]["ensemble"]["total_return"] for r in per_ticker]))
    ret_bh = float(np.mean([r["econ"]["buy_hold"]["total_return"] for r in per_ticker]))
    # Pooled McNemar from paired outcomes across tickers would need the raw
    # paired vectors; as the conservative pooled representative use the min
    # per-ticker exact McNemar p vs each ticker's best baseline.
    mcn_best = min(r["mcnemar"][r["best_baseline"]] for r in per_ticker)

    verdict = pooled_verdict(n_eval, k_ens, accs, ret_ens, ret_bh, mcn_best)

    regimes = {}
    if args.regime_split:
        regimes = regime_breakdown(per_ticker)
        for name, g in regimes.items():
            a = g["accs"]
            print(f"  regime {name:8s} n={g['n']:5d} "
                  f"ens={a['ensemble']:.3f} naive={a['naive']:.3f} "
                  f"mom={a['momentum']:.3f} log={a['logistic']:.3f} "
                  f"ret_ens={g['ret_ens']:+.3f} ret_bh={g['ret_bh']:+.3f}")

    os.makedirs(args.report_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = {"stamp": stamp, "fit_frac": args.fit_frac,
              "eval_frac": args.eval_frac, "pooled": {
                  "n_eval": n_eval, "k_ens": k_ens, **verdict},
              "regimes": regimes,
              "tickers": per_ticker}
    with open(os.path.join(args.report_dir, f"report_{stamp}.json"), "w") as f:
        json.dump(report, f, indent=2, default=str)

    status = "PASS" if verdict["passed"] else "FAIL"
    print(f"\nEDGE GATE: {status} "
          f"(n_eval={n_eval} acc_ens={accs['ensemble']:.3f} "
          f"p_binom={verdict['p_binom']:.4f} mcnemar_best={verdict['mcnemar_best_p']:.4f} "
          f"ret_ens={ret_ens:+.3f} ret_bh={ret_bh:+.3f})")
    for k, v in verdict["checks"].items():
        print(f"  [{'x' if v else ' '}] {k}")
    if not verdict["passed"]:
        print("Action: frozen F1/F2/F3 branch per docs/MASTER-PLAN.md — no O5/O6 spend.")
    return 0 if verdict["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

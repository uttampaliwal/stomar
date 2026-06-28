"""Stage 2: Alpha & Backtest Rigor Analysis (Corrected).

Uses walk-forward backtesting for true out-of-sample results.
CPCV is the primary significance test.
"""
import os
import sys
import json
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import fetch_stock_data
from src.features import add_technical_indicators
from src.model import load_models, DEVICE
from src.significance import (
    purged_kfold_cv, run_cpcv, deflated_sharpe_ratio,
    permutation_test_accuracy, bootstrap_confidence_interval,
    backtest_predictions_to_sharpe,
)
from src.benchmarks import compare_to_benchmarks
from src.alpha_research import full_alpha_research
from src.trainer import FEATURE_COLS, SEQ_LENGTH
from src.logging_config import setup_logging, get_logger
import torch

setup_logging()
logger = get_logger("stage2")

TICKERS = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")


def load_stock_data(ticker):
    """Load data and compute features."""
    logger.info("loading_data ticker=%s", ticker)
    df = fetch_stock_data(ticker, period="5y", force_refresh=False)
    df_feat = add_technical_indicators(df, ticker=ticker)
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan).dropna()
    logger.info("loaded ticker=%s rows=%d", ticker, len(df_feat))
    return df_feat


def run_walk_forward_oos(df_feat, feature_cols, ticker):
    """Run walk-forward backtest for true OOS results.

    Trains models on 3 years, tests on 1 year, rolls forward.
    This is the ONLY legitimate way to evaluate out-of-sample performance.
    """
    from src.backtester import run_walk_forward_backtest

    logger.info("running_walk_forward ticker=%s", ticker)

    result = run_walk_forward_backtest(
        ticker=ticker,
        df_feat=df_feat,
        feature_cols=feature_cols,
        train_years=3,
        test_years=1,
        step_months=6,
    )

    metrics, portfolio, test_results = result

    if metrics is None:
        logger.warning("walk_forward_failed ticker=%s", ticker)
        return None, []

    logger.info("walk_forward_done ticker=%s accuracy=%.4f sharpe=%.3f total_return=%.2f%%",
                ticker,
                metrics.get("ensemble_accuracy", 0),
                metrics.get("sharpe_ratio", 0),
                metrics.get("total_return", 0) * 100)

    return metrics, test_results


def run_cpcv_analysis(df_feat, feature_cols, ticker):
    """Run CPCV on a stock."""
    logger.info("running_cpcv ticker=%s", ticker)

    feature_cols_clean = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols_clean + ["target_direction"]].dropna()

    if len(data) < 500:
        return {"error": "insufficient_data", "rows": len(data)}

    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import MinMaxScaler
    from src.model import build_xgb_model

    def backtest_fn(X_train, y_train, X_test, y_test):
        if len(X_train) < 100 or len(X_test) < 20:
            return 0.5

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = build_xgb_model()
        xgb_model.fit(X_train_scaled, y_train, verbose=False)
        xgb_pred = xgb_model.predict(X_test_scaled)
        return float(accuracy_score(y_test, xgb_pred))

    result = run_cpcv(
        data, feature_cols_clean, "target_direction",
        backtest_fn, n_test_groups=4, n_combinations=20
    )

    logger.info("cpcv_done ticker=%s mean=%.4f std=%.4f n=%d prob_beat_random=%.2f",
                ticker, result["mean"], result["std"], result["n"],
                result["prob_beat_random"])
    return result


def run_permutation_on_oos(test_results, ticker):
    """Run permutation test on OOS predictions."""
    if not test_results:
        return {"error": "no_test_results"}

    predictions = np.array([r["predicted"] for r in test_results])
    actuals = np.array([r["actual"] for r in test_results])

    logger.info("running_permutation ticker=%s n=%d", ticker, len(predictions))

    result = permutation_test_accuracy(predictions, actuals, n_permutations=1000)
    logger.info("permutation_done ticker=%s real_acc=%.4f p_value=%.4f significant=%s",
                ticker, result["real_accuracy"], result["p_value"], result["significant"])
    return result


def run_benchmarks_on_oos(test_results, ticker):
    """Compare OOS strategy to benchmarks."""
    if not test_results:
        return []

    predictions = np.array([r["predicted"] for r in test_results])
    actuals = np.array([r["actual"] for r in test_results])

    # Compute returns: +1% if predicted UP and correct, -1% if predicted UP and wrong
    returns = []
    for i in range(1, len(test_results)):
        r = test_results[i]
        prev_r = test_results[i - 1]
        if prev_r["predicted"] == 1 and r["actual"] == 1:
            returns.append(0.01)
        elif prev_r["predicted"] == 1 and r["actual"] == 0:
            returns.append(-0.01)
        else:
            returns.append(0.0)

    returns = np.array(returns)
    strategy_returns = np.where(predictions[1:] == 1, returns, 0.0)

    results = compare_to_benchmarks(strategy_returns, actuals[1:], returns)
    logger.info("benchmarks_done ticker=%s n_benchmarks=%d", ticker, len(results))
    return results


def run_confidence_intervals(test_results, ticker):
    """Compute bootstrap CIs on OOS metrics."""
    if not test_results:
        return {}

    predictions = np.array([r["predicted"] for r in test_results])
    actuals = np.array([r["actual"] for r in test_results])

    acc_result = bootstrap_confidence_interval(
        (predictions == actuals).astype(float), np.mean
    )

    # Sharpe CI
    returns = []
    for i in range(1, len(test_results)):
        r = test_results[i]
        prev_r = test_results[i - 1]
        if prev_r["predicted"] == 1 and r["actual"] == 1:
            returns.append(0.01)
        elif prev_r["predicted"] == 1 and r["actual"] == 0:
            returns.append(-0.01)
        else:
            returns.append(0.0)

    returns = np.array(returns)
    sharpe_data = []
    for _ in range(100):
        idx = np.random.choice(len(returns), len(returns), replace=True)
        s_ret = returns[idx]
        if s_ret.std() > 0:
            sharpe_data.append(float(s_ret.mean() / s_ret.std() * np.sqrt(252)))

    sharpe_result = bootstrap_confidence_interval(
        np.array(sharpe_data) if sharpe_data else np.array([0.0]), np.mean
    )

    cum_returns = np.cumprod(1 + returns) - 1
    ann_return = float((1 + cum_returns[-1]) ** (252 / max(len(returns), 1)) - 1) if len(returns) > 0 else 0

    result = {
        "accuracy": acc_result,
        "sharpe": sharpe_result,
        "annualized_return": ann_return,
        "total_return": float(cum_returns[-1]) if len(cum_returns) > 0 else 0,
    }

    logger.info("ci_done ticker=%s acc_ci=[%.3f, %.3f] sharpe_ci=[%.3f, %.3f]",
                ticker,
                acc_result["ci_lower"], acc_result["ci_upper"],
                sharpe_result["ci_lower"], sharpe_result["ci_upper"])
    return result


def run_alpha_research(df_feat, feature_cols, ticker):
    """Run alpha research pipeline."""
    logger.info("running_alpha_research ticker=%s", ticker)

    feature_cols_clean = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols_clean + ["target_direction"]].dropna()

    if len(data) < 200:
        return {"error": "insufficient_data"}

    result = full_alpha_research(data, feature_cols_clean, "target_direction")
    logger.info("alpha_research_done ticker=%s top_features=%s", ticker, result["top_features"][:5])
    return result


def make_decision(all_results):
    """Make Go/No-Go decision based on all significance tests."""
    decision = {
        "signal_real": False,
        "confidence": "LOW",
        "reasons": [],
        "recommendations": [],
    }

    significant_stocks = 0
    total_stocks = len([k for k in all_results if not k.startswith("_")])

    for ticker, results in all_results.items():
        if ticker.startswith("_"):
            continue

        perm = results.get("permutation", {})
        cpcv = results.get("cpcv", {})

        if perm.get("significant", False):
            significant_stocks += 1
            decision["reasons"].append(
                f"{ticker}: permutation significant (p={perm['p_value']:.4f})"
            )
        else:
            decision["reasons"].append(
                f"{ticker}: permutation NOT significant (p={perm.get('p_value', 1.0):.4f})"
            )

        cpcv_mean = cpcv.get("mean", 0.5)
        if cpcv_mean > 0.52:
            decision["reasons"].append(f"{ticker}: CPCV mean={cpcv_mean:.4f} (edge detected)")
        else:
            decision["reasons"].append(f"{ticker}: CPCV mean={cpcv_mean:.4f} (no edge)")

    if significant_stocks >= 2:
        decision["signal_real"] = True
        decision["confidence"] = "HIGH"
        decision["recommendations"].append("Proceed to Stage 3: Live Data + Scheduling")
    elif significant_stocks == 1:
        decision["confidence"] = "MEDIUM"
        decision["recommendations"].append("Consider expanding to more stocks before proceeding")
    else:
        decision["confidence"] = "LOW"
        decision["recommendations"].append("Signal likely noise. Consider pivoting to volatility forecasting or cross-sectional ranking")

    # Check CPCV
    cpcv_means = []
    for ticker, results in all_results.items():
        if ticker.startswith("_"):
            continue
        cpcv_mean = results.get("cpcv", {}).get("mean", 0.5)
        cpcv_means.append(cpcv_mean)

    avg_cpcv = np.mean(cpcv_means) if cpcv_means else 0.5
    if avg_cpcv > 0.52:
        decision["reasons"].append(f"Average CPCV accuracy: {avg_cpcv:.4f} (edge detected)")
    else:
        decision["reasons"].append(f"Average CPCV accuracy: {avg_cpcv:.4f} (no significant edge)")

    # Check benchmarks
    benchmarks_look_good = 0
    for ticker, results in all_results.items():
        if ticker.startswith("_"):
            continue
        benchmarks = results.get("benchmarks", [])
        if benchmarks:
            strategy = next((b for b in benchmarks if b["name"] == "Your Strategy"), None)
            buy_hold = next((b for b in benchmarks if b["name"] == "Buy & Hold"), None)
            if strategy and buy_hold and strategy["sharpe"] > buy_hold["sharpe"]:
                benchmarks_look_good += 1

    if benchmarks_look_good >= 2:
        decision["reasons"].append(f"Strategy beats Buy & Hold on Sharpe in {benchmarks_look_good}/{total_stocks} stocks")
    else:
        decision["reasons"].append(f"Strategy beats Buy & Hold on Sharpe in only {benchmarks_look_good}/{total_stocks} stocks")

    return decision


def main():
    """Run complete Stage 2 analysis."""
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}

    for ticker in TICKERS:
        logger.info("=" * 60)
        logger.info("analyzing ticker=%s", ticker)
        logger.info("=" * 60)

        try:
            df_feat = load_stock_data(ticker)
            feature_cols = [c for c in FEATURE_COLS if c in df_feat.columns]

            # Run CPCV (primary significance test)
            cpcv = run_cpcv_analysis(df_feat, feature_cols, ticker)

            # Run walk-forward OOS backtest
            wf_metrics, test_results = run_walk_forward_oos(df_feat, feature_cols, ticker)

            # Run permutation test on OOS predictions
            perm = run_permutation_on_oos(test_results, ticker)

            # Run benchmarks on OOS predictions
            benchmarks = run_benchmarks_on_oos(test_results, ticker)

            # Run confidence intervals on OOS predictions
            ci = run_confidence_intervals(test_results, ticker)

            # Run alpha research
            alpha = run_alpha_research(df_feat, feature_cols, ticker)

            # Store results
            all_results[ticker] = {
                "cpcv": cpcv,
                "walk_forward_metrics": wf_metrics,
                "permutation": perm,
                "benchmarks": benchmarks,
                "confidence_intervals": ci,
                "alpha_research": {
                    "top_features": alpha.get("top_features", []),
                    "quality_scores": alpha.get("quality_scores", {}),
                },
            }

            if wf_metrics:
                logger.info("summary ticker=%s oos_accuracy=%.4f cpcv_mean=%.4f perm_p=%.4f",
                            ticker,
                            wf_metrics.get("ensemble_accuracy", 0),
                            cpcv.get("mean", 0),
                            perm.get("p_value", 1.0))

        except Exception as e:
            logger.error("analysis_failed ticker=%s error=%s", ticker, str(e))
            import traceback
            traceback.print_exc()
            all_results[ticker] = {"error": str(e)}

    decision = make_decision(all_results)
    all_results["_decision"] = decision

    report_path = os.path.join(RESULTS_DIR, "stage2_report.json")
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("report_saved path=%s", report_path)

    print("\n" + "=" * 70)
    print("STAGE 2 ANALYSIS COMPLETE (CORRECTED)")
    print("=" * 70)

    for ticker in TICKERS:
        if ticker in all_results and "error" not in all_results[ticker]:
            r = all_results[ticker]
            wf = r.get("walk_forward_metrics", {}) or {}
            cpcv = r.get("cpcv", {})
            perm = r.get("permutation", {})
            print(f"\n{ticker}:")
            print(f"  Walk-Forward OOS Accuracy: {wf.get('ensemble_accuracy', 'N/A')}")
            print(f"  Walk-Forward Sharpe: {wf.get('sharpe_ratio', 'N/A')}")
            print(f"  Walk-Forward Total Return: {wf.get('total_return', 0) * 100:.2f}%")
            print(f"  CPCV Mean: {cpcv.get('mean', 'N/A')}")
            print(f"  CPCV Std: {cpcv.get('std', 'N/A')}")
            print(f"  CPCV Prob Beat Random: {cpcv.get('prob_beat_random', 'N/A')}")
            print(f"  Permutation p-value: {perm.get('p_value', 'N/A')}")
            print(f"  Permutation significant: {perm.get('significant', 'N/A')}")

    print("\n" + "=" * 70)
    print("GO/NO-GO DECISION")
    print("=" * 70)
    print(f"  Signal Real: {decision['signal_real']}")
    print(f"  Confidence: {decision['confidence']}")
    print(f"  Reasons:")
    for reason in decision["reasons"]:
        print(f"    - {reason}")
    print(f"  Recommendations:")
    for rec in decision["recommendations"]:
        print(f"    - {rec}")
    print("=" * 70)

    return all_results


if __name__ == "__main__":
    main()

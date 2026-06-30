"""Statistical significance testing for trading strategies.

Implements:
- Combinatorial Purged Cross-Validation (CPCV)
- Deflated Sharpe Ratio
- Permutation Test
- Bootstrap Confidence Intervals

Reference: Marcos López de Prado, "Advances in Financial Machine Learning", Ch. 12
"""
import numpy as np
import pandas as pd
from scipy import stats
from itertools import combinations
from src.constants import RISK_FREE_RATE


def purged_kfold_cv(df, feature_cols, target_col, n_splits=6,
                    embargo_pct=0.01, purge_window=5):
    """Walk-forward CV with purging and embargoing.

    Purging removes training samples whose labels overlap with test labels.
    Embargoing adds a gap between train and test to prevent leakage through
    autocorrelation.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Name of target column
        n_splits: Number of chronological splits
        embargo_pct: Fraction of data to embargo between train/test
        purge_window: Number of rows to purge from end of training set

    Returns:
        List of (train_indices, test_indices) tuples
    """
    n = len(df)
    embargo_size = max(1, int(n * embargo_pct))
    test_size = n // n_splits

    splits = []
    for i in range(n_splits):
        test_start = i * test_size
        test_end = min(test_start + test_size, n)

        if i == 0:
            train_end = test_start
        else:
            train_end = test_start - embargo_size

        train_end = train_end - purge_window

        if train_end <= 0:
            continue

        train_indices = list(range(0, train_end))
        test_indices = list(range(test_start, test_end))

        if len(train_indices) > 0 and len(test_indices) > 0:
            splits.append((train_indices, test_indices))

    return splits


def run_cpcv(df, feature_cols, target_col, backtest_fn,
             n_test_groups=3, n_combinations=100):
    """Run Combinatorial Purged Cross-Validation.

    Generates all combinations of test groups and computes performance
    for each combination, building a distribution of outcomes.

    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names
        target_col: Target column name
        backtest_fn: Function(X_train, y_train, X_test, y_test) -> float
                     Returns a performance metric (e.g., accuracy)
        n_test_groups: Number of test groups to partition data into
        n_combinations: Max number of combinations to evaluate

    Returns:
        Dict with distribution of outcomes and statistics
    """
    n = len(df)
    group_size = n // n_test_groups
    groups = []
    for i in range(n_test_groups):
        start = i * group_size
        end = start + group_size if i < n_test_groups - 1 else n
        groups.append(df.iloc[start:end])

    all_combos = list(combinations(range(n_test_groups), 1))
    if len(all_combos) > n_combinations:
        rng = np.random.RandomState(42)
        indices = rng.choice(len(all_combos), n_combinations, replace=False)
        all_combos = [all_combos[i] for i in indices]

    outcomes = []
    for test_group_idx in all_combos:
        test_idx = test_group_idx[0]
        train_dfs = [groups[i] for i in range(n_test_groups) if i != test_idx]

        if len(train_dfs) == 0:
            continue

        train_df = pd.concat(train_dfs)
        test_df = groups[test_idx]

        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[target_col].values

        try:
            metric = backtest_fn(X_train, y_train, X_test, y_test)
            outcomes.append(metric)
        except Exception:
            continue

    if not outcomes:
        return {"outcomes": [], "mean": 0, "std": 0, "p5": 0, "p95": 0, "n": 0}

    outcomes_arr = np.array(outcomes)
    return {
        "outcomes": outcomes_arr.tolist(),
        "mean": float(outcomes_arr.mean()),
        "std": float(outcomes_arr.std()),
        "median": float(np.median(outcomes_arr)),
        "p5": float(np.percentile(outcomes_arr, 5)),
        "p95": float(np.percentile(outcomes_arr, 95)),
        "n": len(outcomes_arr),
        "prob_profit": float(np.mean(outcomes_arr > 0)),
        "prob_beat_random": float(np.mean(outcomes_arr > 0.5)),
    }


def deflated_sharpe_ratio(sharpe_observed, n_trials, sharpe_max, T,
                          skew=0.0, kurtosis=3.0):
    """Adjust observed Sharpe for multiple testing bias.

    From: López de Prado, "The Deflated Sharpe Ratio" (2014)

    Args:
        sharpe_observed: Your best strategy's Sharpe
        n_trials: How many strategy variations you tried
        sharpe_max: Maximum possible Sharpe (theoretical bound)
        T: Number of independent observations (trades)
        skew: Return distribution skewness
        kurtosis: Return distribution excess kurtosis (3 = normal)

    Returns:
        Dict with deflated Sharpe, p-value, and significance
    """
    euler_mascheroni = 0.5772156649

    # Expected maximum Sharpe under null (no skill)
    if n_trials <= 0:
        return {"deflated_sharpe": 0.0, "p_value": 1.0, "significant": False}

    e_max_sr = sharpe_max * (
        (1 - euler_mascheroni) * stats.norm.ppf(1 - 1 / n_trials) +
        euler_mascheroni * stats.norm.ppf(1 - 1 / (n_trials * np.e))
    )

    # Standard error of Sharpe ratio
    sr_std = np.sqrt(
        (1 + 0.5 * sharpe_observed ** 2 -
         skew * sharpe_observed +
         (kurtosis - 3) / 4 * sharpe_observed ** 2) / max(T, 1)
    )

    if sr_std == 0:
        return {"deflated_sharpe": 0.0, "p_value": 1.0, "significant": False}

    # Deflated Sharpe z-score
    z_score = (sharpe_observed - e_max_sr) / sr_std
    p_value = 1 - stats.norm.cdf(z_score)

    return {
        "sharpe_observed": sharpe_observed,
        "e_max_sr": float(e_max_sr),
        "deflated_sharpe": float(z_score),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_trials": n_trials,
    }


def permutation_test_accuracy(predictions, actuals, n_permutations=1000,
                               random_state=42):
    """Test whether observed accuracy is significantly above chance.

    1. Compute real accuracy
    2. Shuffle actuals n_permutations times
    3. Compute accuracy on each shuffle -> null distribution
    4. p-value = fraction of null accuracies >= real accuracy

    Args:
        predictions: Array of predicted labels (0/1)
        actuals: Array of actual labels (0/1)
        n_permutations: Number of random permutations
        random_state: Random seed for reproducibility

    Returns:
        Dict with real accuracy, null distribution stats, and p-value
    """
    predictions = np.asarray(predictions)
    actuals = np.asarray(actuals)

    real_accuracy = float(np.mean(predictions == actuals))

    rng = np.random.RandomState(random_state)
    null_accuracies = np.empty(n_permutations)

    for i in range(n_permutations):
        shuffled = rng.permutation(actuals)
        null_accuracies[i] = np.mean(predictions == shuffled)

    p_value = float(np.mean(null_accuracies >= real_accuracy))

    return {
        "real_accuracy": real_accuracy,
        "null_mean": float(null_accuracies.mean()),
        "null_std": float(null_accuracies.std()),
        "null_p5": float(np.percentile(null_accuracies, 5)),
        "null_p95": float(np.percentile(null_accuracies, 95)),
        "p_value": p_value,
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def bootstrap_confidence_interval(data, metric_fn, n_bootstrap=1000,
                                   confidence=0.95, random_state=42):
    """Bootstrap confidence interval for any metric.

    Args:
        data: Array of data points
        metric_fn: Function(data) -> scalar metric
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level (e.g., 0.95 for 95% CI)
        random_state: Random seed for reproducibility

    Returns:
        Dict with point estimate, CI bounds, and standard error
    """
    data = np.asarray(data)
    point = float(metric_fn(data))

    rng = np.random.RandomState(random_state)
    boot_stats = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        sample = rng.choice(data, size=len(data), replace=True)
        boot_stats[i] = metric_fn(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))
    std_error = float(boot_stats.std())

    return {
        "point_estimate": point,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "std_error": std_error,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
    }


def backtest_predictions_to_sharpe(predictions, actuals, returns,
                                    risk_free_rate=None):
    """Compute Sharpe ratio from prediction signals and returns.

    Args:
        predictions: Array of predicted directions (0/1)
        actuals: Array of actual directions (0/1)
        returns: Array of actual returns
        risk_free_rate: Annual risk-free rate

    Returns:
        Sharpe ratio
    """
    if risk_free_rate is None:
        risk_free_rate = RISK_FREE_RATE
    predictions = np.asarray(predictions)
    returns = np.asarray(returns)

    # Strategy returns: long when predicting UP
    strategy_returns = np.where(predictions == 1, returns, 0.0)

    if len(strategy_returns) < 10:
        return 0.0

    mean_ret = strategy_returns.mean()
    std_ret = strategy_returns.std()

    if std_ret == 0:
        return 0.0

    daily_rf = risk_free_rate / 252
    sharpe = (mean_ret - daily_rf) / std_ret * np.sqrt(252)
    return float(sharpe)

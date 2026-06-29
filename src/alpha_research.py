"""Alpha research pipeline.

Hypothesis testing per signal, decay analysis, and signal ranking
to identify which features actually predict returns.
"""
import numpy as np
import pandas as pd
from scipy import stats


def test_hypothesis(df, feature_col, target_col, n_quantiles=5):
    """Test whether a feature predicts the target.

    Groups data by feature quantiles and tests if mean target
    differs significantly across groups.

    Args:
        df: DataFrame with features and target
        feature_col: Feature column name
        target_col: Target column name
        n_quantiles: Number of quantile groups

    Returns:
        Dict with hypothesis test results
    """
    data = df[[feature_col, target_col]].dropna()
    if len(data) < n_quantiles * 2:
        return {
            "feature": feature_col,
            "f_stat": 0.0, "p_value": 1.0,
            "significant": False, "effect_size": 0.0,
            "direction": 0, "n_groups": 0,
        }

    try:
        data["quantile"] = pd.qcut(data[feature_col], n_quantiles,
                                    labels=False, duplicates="drop")
    except ValueError:
        return {
            "feature": feature_col,
            "f_stat": 0.0, "p_value": 1.0,
            "significant": False, "effect_size": 0.0,
            "direction": 0, "n_groups": 0,
        }

    groups = [group[target_col].values for _, group in data.groupby("quantile")
              if len(group[target_col].values) >= 2]

    if len(groups) < 2:
        return {
            "feature": feature_col,
            "f_stat": 0.0, "p_value": 1.0,
            "significant": False, "effect_size": 0.0,
            "direction": 0, "n_groups": 0,
        }

    f_stat, p_value = stats.f_oneway(*groups)

    # Effect size (eta-squared)
    grand_mean = data[target_col].mean()
    ss_between = sum(len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups)
    ss_total = np.sum((data[target_col].values - grand_mean) ** 2)
    eta_sq = ss_between / ss_total if ss_total > 0 else 0

    # Direction: does higher feature value predict higher target?
    group_means = [np.mean(g) for g in groups]
    direction = 1 if group_means[-1] > group_means[0] else -1

    return {
        "feature": feature_col,
        "f_stat": float(f_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "effect_size": float(eta_sq),
        "direction": direction,
        "n_groups": len(groups),
    }


def decay_analysis(df, feature_col, target_col, max_lag=20):
    """Analyze signal decay over time.

    Tests predictive power at different forward lags.

    Args:
        df: DataFrame with features and target
        feature_col: Feature column name
        target_col: Target column name
        max_lag: Maximum forward lag to test

    Returns:
        Dict with decay curve
    """
    results = {"lags": [], "correlations": [], "p_values": []}

    for lag in range(0, max_lag + 1):
        shifted = df[target_col].shift(-lag)
        valid_mask = shifted.notna() & df[feature_col].notna()
        if valid_mask.sum() < 10:
            break

        corr, p_val = stats.spearmanr(
            df.loc[valid_mask, feature_col],
            shifted[valid_mask]
        )

        results["lags"].append(lag)
        results["correlations"].append(float(corr))
        results["p_values"].append(float(p_val))

    if results["correlations"]:
        abs_corrs = [abs(c) for c in results["correlations"]]
        peak_lag = results["lags"][np.argmax(abs_corrs)]
        half_life = None
        peak_corr = max(abs_corrs)

        for i, corr in enumerate(abs_corrs):
            if corr < peak_corr / 2:
                half_life = results["lags"][i]
                break

        results["peak_lag"] = peak_lag
        results["half_life"] = half_life
        results["n_lags"] = len(results["lags"])
    else:
        results["peak_lag"] = 0
        results["half_life"] = None
        results["n_lags"] = 0

    return results


def rank_signals(df, feature_cols, target_col):
    """Rank features by predictive power.

    Args:
        df: DataFrame with features and target
        feature_cols: List of feature column names
        target_col: Target column name

    Returns:
        List of feature ranking dicts, sorted by p_value
    """
    rankings = []
    for feature in feature_cols:
        result = test_hypothesis(df, feature, target_col)
        rankings.append(result)

    rankings.sort(key=lambda x: x["p_value"])
    return rankings


def compute_ic_series(df, feature_col, target_col, window=60):
    """Compute rolling Information Coefficient (rank correlation).

    Args:
        df: DataFrame with features and target
        feature_col: Feature column name
        target_col: Target column name
        window: Rolling window size

    Returns:
        Series of rolling IC values
    """
    valid_data = df[[feature_col, target_col]].dropna()
    if len(valid_data) < window:
        return pd.Series(dtype=float)

    ranks_feat = valid_data[feature_col].rolling(window).rank()
    ranks_tgt = valid_data[target_col].rolling(window).rank()
    ic_values = ranks_tgt.corr(ranks_feat)
    return ic_values


def signal_autocorrelation(df, feature_col, max_lag=10):
    """Compute autocorrelation of a signal.

    Args:
        df: DataFrame with features
        feature_col: Feature column name
        max_lag: Maximum lag to test

    Returns:
        Dict with autocorrelation values
    """
    series = df[feature_col].dropna()
    if len(series) < max_lag + 2:
        return {"lags": [], "autocorrelations": []}

    lags = list(range(1, max_lag + 1))
    autocorrs = []

    for lag in lags:
        if len(series) <= lag:
            break
        corr = series.corr(series.shift(lag))
        autocorrs.append(float(corr) if not np.isnan(corr) else 0.0)

    return {"lags": lags, "autocorrelations": autocorrs}


def compute_signal_quality_score(hypothesis_result, decay_result):
    """Compute a quality score for a signal.

    Args:
        hypothesis_result: Output from test_hypothesis
        decay_result: Output from decay_analysis

    Returns:
        Float quality score (0-1)
    """
    # Base score from statistical significance
    if hypothesis_result["p_value"] < 0.01:
        sig_score = 1.0
    elif hypothesis_result["p_value"] < 0.05:
        sig_score = 0.7
    elif hypothesis_result["p_value"] < 0.10:
        sig_score = 0.4
    else:
        sig_score = 0.0

    # Effect size score (max ~0.1 for strong effect)
    effect_score = min(hypothesis_result["effect_size"] / 0.1, 1.0)

    # Decay score: longer half-life = better
    half_life = decay_result.get("half_life")
    if half_life is None:
        decay_score = 0.5
    elif half_life >= 10:
        decay_score = 1.0
    elif half_life >= 5:
        decay_score = 0.7
    else:
        decay_score = 0.3

    return sig_score * 0.4 + effect_score * 0.3 + decay_score * 0.3


def full_alpha_research(df, feature_cols, target_col):
    """Run complete alpha research pipeline.

    Args:
        df: DataFrame with features and target
        feature_cols: Feature column names
        target_col: Target column name

    Returns:
        Dict with rankings, decay curves, and quality scores
    """
    rankings = rank_signals(df, feature_cols, target_col)

    decay_curves = {}
    quality_scores = {}

    for feature in feature_cols:
        hypothesis_result = test_hypothesis(df, feature, target_col)
        decay_result = decay_analysis(df, feature, target_col, max_lag=10)
        decay_curves[feature] = decay_result
        quality_scores[feature] = compute_signal_quality_score(
            hypothesis_result, decay_result
        )

    sorted_features = sorted(quality_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "rankings": rankings,
        "decay_curves": decay_curves,
        "quality_scores": quality_scores,
        "top_features": [f for f, _ in sorted_features[:10]],
        "bottom_features": [f for f, _ in sorted_features[-5:]],
    }

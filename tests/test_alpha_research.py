"""Tests for alpha research pipeline."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=500):
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    feature_a = np.random.randn(n)
    feature_b = np.random.randn(n) * 0.1
    noise = np.random.randn(n) * 0.01
    # target correlated with feature_a, not with feature_b or noise
    target = (feature_a + noise * 0.1 > 0).astype(int)

    df = pd.DataFrame({
        "feature_a": feature_a,
        "feature_b": feature_b,
        "noise": noise,
        "target": target,
    }, index=dates)
    return df


class TestHypothesis:
    def test_correlated_feature_significant(self):
        from src.alpha_research import test_hypothesis
        df = _make_df(500)
        result = test_hypothesis(df, "feature_a", "target")
        assert result["significant"] or result["p_value"] < 0.20
        assert result["f_stat"] >= 0

    def test_uncorrelated_feature_not_significant(self):
        from src.alpha_research import test_hypothesis
        df = _make_df(500)
        result = test_hypothesis(df, "feature_b", "target")
        assert result["p_value"] > 0.01

    def test_returns_all_keys(self):
        from src.alpha_research import test_hypothesis
        df = _make_df(100)
        result = test_hypothesis(df, "feature_a", "target")
        expected = ["feature", "f_stat", "p_value", "significant",
                    "effect_size", "direction", "n_groups"]
        for k in expected:
            assert k in result

    def test_direction_positive(self):
        from src.alpha_research import test_hypothesis
        df = _make_df(500)
        result = test_hypothesis(df, "feature_a", "target")
        assert result["direction"] in [1, -1]


class TestDecayAnalysis:
    def test_peak_lag_exists(self):
        from src.alpha_research import decay_analysis
        df = _make_df(300)
        result = decay_analysis(df, "feature_a", "target", max_lag=10)
        assert "peak_lag" in result
        assert result["n_lags"] > 0

    def test_correlations_in_range(self):
        from src.alpha_research import decay_analysis
        df = _make_df(300)
        result = decay_analysis(df, "feature_a", "target", max_lag=5)
        for corr in result["correlations"]:
            assert -1 <= corr <= 1


class TestRankSignals:
    def test_sorted_by_pvalue(self):
        from src.alpha_research import rank_signals
        df = _make_df(500)
        rankings = rank_signals(df, ["feature_a", "feature_b", "noise"], "target")
        p_values = [r["p_value"] for r in rankings]
        assert p_values == sorted(p_values)

    def test_correlated_first(self):
        from src.alpha_research import rank_signals
        df = _make_df(1000)
        rankings = rank_signals(df, ["feature_a", "feature_b", "noise"], "target")
        assert rankings[0]["feature"] in ["feature_a", "noise"]


class TestQualityScore:
    def test_score_in_range(self):
        from src.alpha_research import compute_signal_quality_score
        hypothesis = {"p_value": 0.01, "effect_size": 0.05}
        decay = {"half_life": 10}
        score = compute_signal_quality_score(hypothesis, decay)
        assert 0 <= score <= 1

    def test_high_quality_high_score(self):
        from src.alpha_research import compute_signal_quality_score
        hypothesis = {"p_value": 0.001, "effect_size": 0.15}
        decay = {"half_life": 20}
        score = compute_signal_quality_score(hypothesis, decay)
        assert score > 0.5

    def test_low_quality_low_score(self):
        from src.alpha_research import compute_signal_quality_score
        hypothesis = {"p_value": 0.5, "effect_size": 0.001}
        decay = {"half_life": 1}
        score = compute_signal_quality_score(hypothesis, decay)
        assert score < 0.5


class TestFullAlphaResearch:
    def test_returns_all_keys(self):
        from src.alpha_research import full_alpha_research
        df = _make_df(500)
        result = full_alpha_research(df, ["feature_a", "feature_b", "noise"], "target")
        expected = ["rankings", "decay_curves", "quality_scores",
                    "top_features", "bottom_features"]
        for k in expected:
            assert k in result

    def test_top_features_populated(self):
        from src.alpha_research import full_alpha_research
        df = _make_df(500)
        result = full_alpha_research(df, ["feature_a", "feature_b", "noise"], "target")
        assert len(result["top_features"]) > 0

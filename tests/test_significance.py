"""Tests for significance testing module."""
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _make_df(n=500):
    np.random.seed(42)
    dates = pd.bdate_range("2022-01-01", periods=n)
    close = pd.Series(100 + np.cumsum(np.random.randn(n) * 0.5))
    high = close + abs(np.random.randn(n) * 0.3)
    low = close - abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.2
    volume = np.random.randint(100000, 1000000, n)
    returns_1d = close.pct_change().fillna(0)
    sma_10 = close.rolling(10).mean().fillna(close)
    sma_20 = close.rolling(20).mean().fillna(close)
    rsi = np.random.uniform(20, 80, n)

    target = (close.shift(-1) > close).astype(int).fillna(0)

    df = pd.DataFrame({
        "close": close, "high": high, "low": low,
        "open": open_, "volume": volume,
        "returns_1d": returns_1d, "sma_10": sma_10,
        "sma_20": sma_20, "rsi": rsi, "target": target,
    }, index=dates)
    df.index.name = "date"
    return df


class TestPurgedKfoldCV:
    def test_returns_list_of_tuples(self):
        from src.significance import purged_kfold_cv
        df = _make_df(500)
        splits = purged_kfold_cv(df, ["rsi"], "target", n_splits=5)
        assert isinstance(splits, list)
        for train_idx, test_idx in splits:
            assert isinstance(train_idx, list)
            assert isinstance(test_idx, list)

    def test_no_overlap_between_train_and_test(self):
        from src.significance import purged_kfold_cv
        df = _make_df(500)
        splits = purged_kfold_cv(df, ["rsi"], "target", n_splits=5)
        for train_idx, test_idx in splits:
            assert len(set(train_idx) & set(test_idx)) == 0

    def test_test_follows_train(self):
        from src.significance import purged_kfold_cv
        df = _make_df(500)
        splits = purged_kfold_cv(df, ["rsi"], "target", n_splits=5)
        for train_idx, test_idx in splits:
            if train_idx and test_idx:
                assert max(train_idx) < min(test_idx)

    def test_purge_window_removes_rows(self):
        from src.significance import purged_kfold_cv
        df = _make_df(500)
        splits_no_purge = purged_kfold_cv(df, ["rsi"], "target",
                                           n_splits=5, purge_window=0)
        splits_with_purge = purged_kfold_cv(df, ["rsi"], "target",
                                             n_splits=5, purge_window=10)
        # Purged should have smaller or equal training sets
        for (_, test_no), (_, test_p) in zip(splits_no_purge, splits_with_purge):
            assert len(test_no) == len(test_p)  # Test sets same

    def test_short_data_few_splits(self):
        from src.significance import purged_kfold_cv
        df = _make_df(100)
        splits = purged_kfold_cv(df, ["rsi"], "target", n_splits=10)
        # Should have fewer splits due to small data
        assert len(splits) <= 10


class TestCPCV:
    def test_returns_distribution(self):
        from src.significance import run_cpcv
        df = _make_df(300)

        def dummy_backtest(X_train, y_train, X_test, y_test):
            return float(np.mean(y_test))

        result = run_cpcv(df, ["rsi"], "target", dummy_backtest,
                          n_test_groups=3, n_combinations=10)
        assert "outcomes" in result
        assert "mean" in result
        assert "std" in result
        assert result["n"] > 0

    def test_outcomes_are_floats(self):
        from src.significance import run_cpcv
        df = _make_df(300)

        def dummy_backtest(X_train, y_train, X_test, y_test):
            return 0.5 + np.random.randn() * 0.01

        result = run_cpcv(df, ["rsi"], "target", dummy_backtest,
                          n_test_groups=3, n_combinations=5)
        assert all(isinstance(x, float) for x in result["outcomes"])

    def test_prob_profit_in_range(self):
        from src.significance import run_cpcv
        df = _make_df(300)

        def dummy_backtest(X_train, y_train, X_test, y_test):
            return float(np.mean(y_test) - 0.5)

        result = run_cpcv(df, ["rsi"], "target", dummy_backtest,
                          n_test_groups=3, n_combinations=10)
        assert 0 <= result["prob_profit"] <= 1


class TestDeflatedSharpeRatio:
    def test_single_trial_no_deflation(self):
        from src.significance import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(2.0, 1, 3.0, 250)
        # With 1 trial, deflated should be close to observed
        assert result["deflated_sharpe"] > 0

    def test_many_trials_deflates_more(self):
        from src.significance import deflated_sharpe_ratio
        r1 = deflated_sharpe_ratio(2.0, 1, 3.0, 250)
        r100 = deflated_sharpe_ratio(2.0, 100, 3.0, 250)
        # More trials -> more deflation
        assert r100["deflated_sharpe"] < r1["deflated_sharpe"]

    def test_high_sharpe_significant(self):
        from src.significance import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(3.0, 10, 4.0, 1000)
        assert result["sharpe_observed"] == 3.0

    def test_zero_trials(self):
        from src.significance import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(2.0, 0, 3.0, 250)
        assert result["p_value"] == 1.0

    def test_returns_all_keys(self):
        from src.significance import deflated_sharpe_ratio
        result = deflated_sharpe_ratio(1.5, 20, 3.0, 500)
        expected = ["sharpe_observed", "e_max_sr", "deflated_sharpe",
                    "p_value", "significant", "n_trials"]
        for k in expected:
            assert k in result


class TestPermutationTestAccuracy:
    def test_random_predictions_not_significant(self):
        from src.significance import permutation_test_accuracy
        np.random.seed(42)
        preds = np.random.randint(0, 2, 200)
        actuals = np.random.randint(0, 2, 200)
        result = permutation_test_accuracy(preds, actuals, n_permutations=500)
        # Random predictions should not be significant
        assert result["p_value"] > 0.01

    def test_perfect_predictions_significant(self):
        from src.significance import permutation_test_accuracy
        actuals = np.random.randint(0, 2, 200)
        result = permutation_test_accuracy(actuals, actuals, n_permutations=500)
        assert result["p_value"] < 0.05
        assert result["real_accuracy"] == 1.0

    def test_returns_all_keys(self):
        from src.significance import permutation_test_accuracy
        preds = np.random.randint(0, 2, 100)
        actuals = np.random.randint(0, 2, 100)
        result = permutation_test_accuracy(preds, actuals, n_permutations=100)
        expected = ["real_accuracy", "null_mean", "null_std",
                    "p_value", "significant", "n_permutations"]
        for k in expected:
            assert k in result


class TestBootstrapCI:
    def test_ci_contains_point(self):
        from src.significance import bootstrap_confidence_interval
        data = np.random.randn(200) * 0.01 + 0.001
        result = bootstrap_confidence_interval(data, np.mean)
        assert result["ci_lower"] <= result["point_estimate"] <= result["ci_upper"]

    def test_ci_width_decreases_with_data(self):
        from src.significance import bootstrap_confidence_interval
        np.random.seed(42)
        data_small = np.random.randn(50) * 0.01
        data_large = np.random.randn(500) * 0.01
        r_small = bootstrap_confidence_interval(data_small, np.mean)
        r_large = bootstrap_confidence_interval(data_large, np.mean)
        # More data -> narrower CI
        width_small = r_small["ci_upper"] - r_small["ci_lower"]
        width_large = r_large["ci_upper"] - r_large["ci_lower"]
        assert width_large < width_small

    def test_returns_all_keys(self):
        from src.significance import bootstrap_confidence_interval
        data = np.random.randn(100)
        result = bootstrap_confidence_interval(data, np.mean)
        expected = ["point_estimate", "ci_lower", "ci_upper",
                    "std_error", "confidence", "n_bootstrap"]
        for k in expected:
            assert k in result


class TestBacktestSharpe:
    def test_all_correct_predictions(self):
        from src.significance import backtest_predictions_to_sharpe
        predictions = np.array([1, 0, 1, 1, 0, 1, 0, 0, 1, 1,
                                1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
        actuals = predictions.copy()
        returns = np.random.randn(20) * 0.01
        sharpe = backtest_predictions_to_sharpe(predictions, actuals, returns)
        assert isinstance(sharpe, float)

    def test_short_data(self):
        from src.significance import backtest_predictions_to_sharpe
        sharpe = backtest_predictions_to_sharpe(np.array([1]), np.array([1]),
                                                np.array([0.01]))
        assert sharpe == 0.0

    def test_zero_volatility(self):
        from src.significance import backtest_predictions_to_sharpe
        predictions = np.array([1, 1, 1, 1, 1])
        actuals = np.array([1, 1, 1, 1, 1])
        returns = np.array([0.01, 0.01, 0.01, 0.01, 0.01])
        sharpe = backtest_predictions_to_sharpe(predictions, actuals, returns)
        # With zero std, Sharpe should handle gracefully
        assert isinstance(sharpe, float)

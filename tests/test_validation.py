"""Tests for src/models/validation.py — purged walk-forward CV + OOS metrics."""

import numpy as np
import pandas as pd
import pytest

from src.models.validation import (
    OOSMetrics,
    PurgedGroupTimeSeriesSplit,
    oos_metrics,
    performance_metrics,
    purged_walk_forward,
    strategy_returns_from_signals,
)


@pytest.fixture
def xy():
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({"f1": rng.normal(size=n), "f2": rng.normal(size=n)})
    y = pd.Series(np.where(rng.normal(size=n) > 0, 1, 0))
    return X, y


@pytest.fixture
def daily_groups():
    dates = pd.date_range("2024-01-01", periods=200, freq="B")
    return pd.DataFrame(
        {"t0": dates, "t1": dates + pd.Timedelta(days=5)}
    )


class TestPurgedGroupTimeSeriesSplit:
    def test_horizon_purge_and_embargo(self, xy):
        X, y = xy
        sp = PurgedGroupTimeSeriesSplit(n_splits=5, embargo=10, horizon=5)
        splits = list(sp.split(X, y))
        assert 1 <= len(splits) <= 5
        for tr, te in splits:
            assert len(tr) > 0 and len(te) > 0
            assert tr.max() < te.min(), "future leakage"
            # no train sample whose label window [t, t+horizon) hits test
            ts = te[0]
            assert not set(tr).intersection(range(ts - 5 + 1, ts))
            # embargo: nothing in [test_end, test_end+embargo)
            te_end = te[-1] + 1
            assert not set(tr).intersection(range(te_end, te_end + 10))

    def test_expanding_window(self, xy):
        X, y = xy
        sp = PurgedGroupTimeSeriesSplit(n_splits=4, embargo=0, horizon=1)
        train_sizes = [len(tr) for tr, _ in sp.split(X, y)]
        assert train_sizes == sorted(train_sizes)
        assert train_sizes[0] >= 0

    def test_t1_datetime_purge(self, xy, daily_groups):
        X, y = xy
        sp = PurgedGroupTimeSeriesSplit(
            n_splits=4, embargo=pd.Timedelta(days=10), horizon=5
        )
        dates = daily_groups["t0"]
        for tr, te in sp.split(X, y, daily_groups):
            test_t0 = dates.iloc[te[0]]
            # purge band is [first test t0, last test sample's label end)
            test_t1 = daily_groups["t1"].iloc[te[-1]]
            t1_train = daily_groups["t1"].iloc[tr]
            assert not ((t1_train >= test_t0) & (t1_train < test_t1)).any(), (
                "train sample label overlaps test fold"
            )
            # ~10 business-day embargo
            assert not set(tr).intersection(range(te[-1] + 1, te[-1] + 11))

    def test_numeric_t1_purge(self, xy):
        X, y = xy
        rng = np.random.default_rng(1)
        t1 = pd.Series(np.arange(200) + rng.integers(1, 6, 200))
        groups = pd.DataFrame({"t1": t1})
        sp = PurgedGroupTimeSeriesSplit(n_splits=4, embargo=0, horizon=5)
        for tr, te in sp.split(X, y, groups):
            test_start, test_end = te[0], te[-1] + 1
            t1_train = groups["t1"].iloc[tr]
            assert not ((t1_train >= test_start) & (t1_train < test_end)).any()

    def test_purge_uses_exact_label_end(self):
        """Regression: purge must extend to the last test sample's label end,
        not the t0 step estimate (which under-purged when labels span
        several days)."""
        n = 40
        X = pd.DataFrame({"f1": np.random.default_rng(7).normal(size=n)})
        y = pd.Series(np.random.default_rng(8).normal(size=n))
        t0 = pd.bdate_range("2024-01-01", periods=n).to_series().reset_index(drop=True)
        t1 = t0 + pd.Timedelta(days=5)  # labels span 5 calendar days
        groups = pd.DataFrame({"t0": t0, "t1": t1})
        sp = PurgedGroupTimeSeriesSplit(n_splits=19, embargo=0, horizon=5)
        leaks = 0
        for tr, te in sp.split(X, y, groups):
            band = (t1.iloc[tr] >= t0.iloc[te[0]]) & (t1.iloc[tr] < t1.iloc[te[-1]])
            leaks += int(band.sum())
        assert leaks == 0

    def test_insufficient_samples(self):
        sp = PurgedGroupTimeSeriesSplit(n_splits=5, embargo=10, horizon=5)
        with pytest.raises(ValueError):
            list(sp.split(pd.DataFrame(np.zeros((5, 2))), np.zeros(5)))

    def test_invalid_params(self):
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(n_splits=1)
        with pytest.raises(ValueError):
            PurgedGroupTimeSeriesSplit(n_splits=3, horizon=0)

    def test_get_n_splits(self, xy):
        X, y = xy
        sp = PurgedGroupTimeSeriesSplit(n_splits=4)
        assert sp.get_n_splits(X, y) == 4


class TestPurgedWalkForward:
    def test_oos_predictions_columns(self, xy, daily_groups):
        X, y = xy

        def dumb_model(Xtr, ytr, Xte):
            return np.full(len(Xte), 1), np.full(len(Xte), 0.9)

        res = purged_walk_forward(
            X, y, dumb_model, groups=daily_groups,
            n_splits=4, embargo=pd.Timedelta(days=10), horizon=5, progress=False,
        )
        assert len(res["oos_predictions"]) > 0
        assert list(res["oos_predictions"].columns) == [
            "fold", "position", "y_true", "y_pred", "proba",
        ]
        assert all(f["n_purged"] >= 0 for f in res["folds"])
        assert all(f["test_size"] > 0 for f in res["folds"])
        assert len(res["folds"]) == len(res["oos_predictions"]["fold"].unique())

    def test_purged_count_is_consistent(self, xy):
        X, y = xy
        res = purged_walk_forward(
            X, y, lambda Xtr, ytr, Xte: np.zeros(len(Xte)),
            n_splits=4, embargo=10, horizon=5, progress=False,
        )
        # train_size + n_purged == test fold start position (indices[:test_start]).
        # Fold k (enumerated over yielded folds) starts at (k+1) * fold_size,
        # since degenerate fold 0 (empty train) is skipped.
        fold_size = len(X) // 4
        for f in res["folds"]:
            assert f["n_purged"] >= 0
            assert f["train_size"] + f["n_purged"] == (f["fold"] + 1) * fold_size

    def test_metrics_present(self, xy):
        X, y = xy
        res = purged_walk_forward(
            X, y, lambda Xtr, ytr, Xte: np.zeros(len(Xte)),
            n_splits=3, embargo=5, horizon=5, progress=False,
        )
        assert "sharpe" in res["metrics"]
        assert "max_drawdown" in res["metrics"]


class TestOOSMetrics:
    def test_constant_positive_returns(self):
        from src.trading.simulate import nse_cost_rates
        buy = nse_cost_rates()["buy"]
        df = pd.DataFrame({"y_true": np.full(100, 0.01), "y_pred": np.full(100, 0.9)})
        m = oos_metrics(df)
        # single entry pays the buy rate once (multiplicatively); holding is free
        expected_total = (1.01 / (1 + buy)) * (1.01 ** 99) - 1
        assert m["total_return"] == pytest.approx(expected_total, abs=1e-4)
        assert m["n_periods"] == 100
        assert m["volatility"] < 0.01
        assert m["sharpe"] > 0
        assert isinstance(m["details"], dict)

    def test_balanced_alternation(self):
        from src.trading.simulate import nse_cost_rates
        buy = nse_cost_rates()["buy"]
        rets = np.tile([0.01, -0.01], 100)
        df = pd.DataFrame({"y_true": rets, "y_pred": np.full(200, 0.9)})
        m = oos_metrics(df)
        # long throughout: only the entry pays; 199 remaining bars alternate
        expected_total = (1.01 / (1 + buy)) * (0.99 * 1.01) ** 99 * 0.99 - 1
        assert m["total_return"] == pytest.approx(expected_total, abs=5e-3)
        vol = np.std(rets, ddof=1)
        assert m["volatility"] == pytest.approx(round(vol * np.sqrt(252), 4), abs=1e-2)
        assert m["sharpe"] <= 0

    def test_signed_strategy(self):
        rets = np.tile([0.01, -0.01], 50)
        df = pd.DataFrame({"y_true": rets, "y_pred": np.full(100, 0.9)})
        m_signed = oos_metrics(df, strategy="signed")
        m_dir = oos_metrics(df, strategy="directional")
        # signed always trades: 2*0.9-1 = 0.8 exposure -> lower vol than full long
        assert m_signed["n_periods"] == 100
        assert m_dir["n_periods"] == 100
        assert m_signed["details"]["n_traded"] == 100

    def test_empty_input(self):
        m = oos_metrics(pd.DataFrame())
        assert m["n_periods"] == 0
        assert m["sharpe"] == 0.0

    def test_all_zero_returns(self):
        from src.trading.simulate import nse_cost_rates
        buy = nse_cost_rates()["buy"]
        df = pd.DataFrame({"y_true": np.zeros(50), "y_pred": np.full(50, 0.9)})
        m = oos_metrics(df)
        assert m["total_return"] == pytest.approx(1 / (1 + buy) - 1, abs=1e-4)

    def test_known_drawdown(self):
        from src.trading.simulate import nse_cost_rates
        rates = nse_cost_rates()
        # long when up, flat when down -> daily round trips pay both rates
        df = pd.DataFrame({
            "y_true": np.tile([0.01, -0.01], 50),
            "y_pred": np.array([0.9, 0.1] * 50),
        })
        m = oos_metrics(df)
        expected_total = (
            (1.01 / (1 + rates["buy"])) / (1 + rates["sell"])
        ) ** 50 - 1
        assert m["total_return"] == pytest.approx(expected_total, abs=1e-4)
        # flat periods bleed the sell rate below the local peak
        assert m["max_drawdown"] < 0
        assert m["max_drawdown"] > -0.01


class TestPerformanceMetrics:
    def test_known_sharpe(self):
        rets = np.tile([0.01, -0.01], 100)
        m = performance_metrics(pd.Series(rets))
        assert m["sharpe"] == 0.0
        vol = np.std(rets, ddof=1)
        assert m["volatility"] == pytest.approx(round(vol * np.sqrt(252), 4), abs=1e-9)
        assert m["n_periods"] == 200

    def test_constant_returns_zero_vol(self):
        m = performance_metrics(pd.Series(np.full(50, 0.01)))
        assert m["volatility"] == 0.0
        assert m["sharpe"] == 0.0

    def test_empty(self):
        m = performance_metrics(pd.Series([]))
        assert m["n_periods"] == 0

    def test_returns_rounded(self):
        m = performance_metrics(pd.Series([0.1, 0.05, -0.02, 0.03]))
        for k, v in m.items():
            if isinstance(v, float):
                assert v == round(v, 4)


class TestStrategyReturns:
    def test_no_lookahead_shift(self):
        rets = pd.Series(np.full(10, 0.01))
        sig = pd.Series([0.9] * 5 + [0.1] * 5)
        sr = strategy_returns_from_signals(rets, sig)
        assert sr.iloc[0] == 0.0, "no prior signal -> flat"
        assert sr.iloc[1] == 0.01, "signal at t=0 applied at t=1"
        assert sr.iloc[6] == 0.0, "SELL signal -> flat"

    def test_threshold(self):
        rets = pd.Series(np.full(10, 0.01))
        sig = pd.Series([0.4, 0.6] + [0.1] * 8)
        sr = strategy_returns_from_signals(rets, sig, threshold=0.5)
        assert sr.iloc[1] == 0.0, "0.4 below threshold"
        assert sr.iloc[2] == 0.01, "0.6 above threshold applied next period"


class TestOOSMetricsDataclass:
    def test_defaults(self):
        m = OOSMetrics()
        assert m.sharpe == 0.0
        assert m.details == {}

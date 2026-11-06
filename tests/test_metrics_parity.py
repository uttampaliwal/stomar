"""Empyrical-reloaded parity oracle for hand-rolled performance metrics.

``src/trading/backtester.compute_metrics`` and
``src/trading/portfolio.Portfolio.get_stats`` implement performance metrics
by hand. These tests validate them against the well-established
``empyrical-reloaded`` package (a ``research`` dependency group, NOT a
production dependency):

* Convention-identical metrics (annualised volatility, max drawdown,
  VaR/CVaR, cumulative return) are compared directly against empyrical.
* ``compute_metrics`` ratios (Sharpe/Sortino/Calmar) annualise a
  *geometric* return over calendar-day spans (365.25 days/year), while
  empyrical annualises the arithmetic mean over 252 trading days. Direct
  library comparison is definitionally impossible, so those tests rebuild
  our exact formulas from empyrical *primitives* (``cum_returns_final``,
  ``annual_volatility``, ``downside_risk``, ``max_drawdown``) and assert
  equality up to the documented output rounding of ``compute_metrics``.
* ``Portfolio.get_stats`` Sharpe is arithmetic-based and matches
  ``empyrical.sharpe_ratio(risk_free=RISK_FREE_RATE / 252)`` exactly.

The tests skip automatically when the research group is not installed
(default ``uv run pytest``); run them explicitly with:

    uv run --group research pytest tests/test_metrics_parity.py -v
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

empyrical = pytest.importorskip("empyrical")


def _risk_free_rate():
    from src.core.constants import RISK_FREE_RATE

    return RISK_FREE_RATE


def _daily_rf():
    return _risk_free_rate() / 252


def _equity_frame(n=500, mu=0.0004, sigma=0.012, seed=42, start=100.0):
    """Deterministic synthetic equity frame plus the aligned daily returns.

    The frame mirrors ``compute_metrics``' input contract (``date`` and
    ``equity`` columns); ``returns`` is derived with exactly the same
    ``pct_change().dropna()`` pipeline used inside ``compute_metrics``.
    """
    rng = np.random.default_rng(seed)
    daily = rng.normal(mu, sigma, n)
    dates = pd.bdate_range("2021-01-01", periods=n + 1)
    equity = start * np.cumprod(1 + daily)
    frame = pd.DataFrame({"date": list(dates[1:]), "equity": equity})
    returns = pd.Series(equity, index=dates[1:]).pct_change().dropna()
    return frame, returns


def _trades_list(pnls):
    """Sell-trade records shaped the way ``compute_metrics`` expects."""
    return [{"pnl": pnl} for pnl in pnls]


class TestConventionIdenticalParity:
    """Metrics whose math is the same convention as empyrical's."""

    SEEDS = [
        (7, 0.0004, 0.012),
        (42, 0.0002, 0.016),
        (123, -0.0008, 0.020),  # bearish path, deep drawdown
        (777, 0.0010, 0.008),
    ]

    def test_annualized_volatility_matches_empyrical(self):
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in self.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            expected = empyrical.annual_volatility(returns, annualization=252)
            assert metrics["annualized_volatility"] == pytest.approx(expected, rel=1e-9)

    def test_max_drawdown_matches_empyrical(self):
        # compute_metrics rounds max_drawdown to 2 decimal places (%), so
        # the comparison uses the corresponding rounding slack.
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in self.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            expected = empyrical.max_drawdown(returns) * 100.0
            assert metrics["max_drawdown"] == pytest.approx(expected, abs=6e-3)

    def test_var_and_cvar_95_match_empyrical(self):
        # VaR/CVaR are rounded to 3 decimal places (%).
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in self.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            var_ep = empyrical.value_at_risk(returns, cutoff=0.05) * 100.0
            cvar_ep = empyrical.conditional_value_at_risk(returns, cutoff=0.05) * 100.0
            assert abs(metrics["var_95"] - var_ep) <= 6e-4
            assert abs(metrics["cvar_95"] - cvar_ep) <= 6e-4

    def test_total_return_matches_empyrical_cum_returns_final(self):
        # total_return is returned unrounded, so the tolerance is tight.
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in self.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            expected = empyrical.cum_returns_final(returns, starting_value=1) - 1
            assert metrics["total_return"] == pytest.approx(expected, rel=1e-9)


class TestGeometricRatioRecomposition:
    """Our ratios rebuilt from empyrical primitives.

    compute_metrics annualises the *geometric* return
    ``(1 + total_return) ** (1 / years) - 1`` with ``years`` measured as
    calendar-day span / 365.25, and divides the rf-adjusted result by the
    *annualised* volatility / downside risk. empyrical instead annualises
    the arithmetic mean (252 periods). The formulas below replicate ours
    exactly, sourcing every risk primitive (cumulative return, annual
    vol, downside risk, max drawdown) from empyrical.
    """

    def _add_geometric_header_and_metrics(self, frame, returns, metrics):
        """Compute our geometric annualisation from empyrical primitives."""
        total_return = empyrical.cum_returns_final(returns, starting_value=1) - 1
        first, last = pd.to_datetime(frame["date"]).iloc[0], pd.to_datetime(frame["date"]).iloc[-1]
        span_days = (last - first).days
        n_years = max(span_days / 365.25, 0.01)
        ann_return = (1 + total_return) ** (1 / n_years) - 1

        ann_vol = empyrical.annual_volatility(returns, annualization=252)
        downside_vol = empyrical.downside_risk(
            returns, required_return=_daily_rf(), annualization=252
        )
        max_dd = empyrical.max_drawdown(returns)

        sharpe = (ann_return - _risk_free_rate()) / ann_vol if ann_vol > 0 else 0
        sortino = (ann_return - _risk_free_rate()) / downside_vol if downside_vol > 0 else 0
        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0
        return sharpe, sortino, calmar, ann_return, max_dd

    def test_annualized_return_is_geometric_from_empyrical_cumulative(self):
        from src.trading.backtester import compute_metrics

        frame, returns = _equity_frame(seed=123, mu=-0.0008, sigma=0.020)
        metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
        total_return = empyrical.cum_returns_final(returns, starting_value=1) - 1
        _, _, _, ann_return, _ = self._add_geometric_header_and_metrics(
            frame, returns, metrics
        )
        assert metrics["annualized_return"] == pytest.approx(ann_return, rel=1e-9)
        assert metrics["total_return"] == pytest.approx(total_return, rel=1e-9)

    def test_sharpe_ratio_rebuilds_from_empyrical_primitives(self):
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in TestConventionIdenticalParity.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            sharpe, _, _, _, _ = self._add_geometric_header_and_metrics(
                frame, returns, metrics
            )
            # sharpe_ratio output is rounded to 3 decimals
            assert metrics["sharpe_ratio"] == pytest.approx(sharpe, abs=6e-4)

    def test_sortino_ratio_rebuilds_from_empyrical_downside_risk(self):
        # This is the critical check: our Sortino denominator
        # sqrt(mean(min(r - rf_daily, 0) ** 2)) * sqrt(252) must equal
        # empyrical.downside_risk(required_return=rf/252), i.e. we share the
        # full-sample floored downside convention.
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in TestConventionIdenticalParity.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            _, sortino, _, _, _ = self._add_geometric_header_and_metrics(
                frame, returns, metrics
            )
            assert metrics["sortino_ratio"] == pytest.approx(sortino, abs=6e-4)

    def test_calmar_rebuilds_from_empyrical_max_drawdown(self):
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in TestConventionIdenticalParity.SEEDS:
            frame, returns = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
            _, _, calmar, _, max_dd = self._add_geometric_header_and_metrics(
                frame, returns, metrics
            )
            assert max_dd < 0, "test series must contain a drawdown for Calmar parity"
            assert metrics["calmar_ratio"] == pytest.approx(calmar, abs=6e-4)


class TestPortfolioSharpeMatchesEmpyrical:
    """Portfolio.get_stats Sharpe is arithmetic-based and matches empyrical."""

    def _driven_portfolio(self):
        from src.trading.portfolio import Portfolio

        portfolio = Portfolio(initial_capital=100_000)
        dates = pd.bdate_range("2024-05-01", periods=8).to_list()
        portfolio.buy("TCS", 3900, 20, dates[0])
        portfolio.buy("INFY", 1550, 50, dates[1])
        portfolio.sell("TCS", 4200, 20, dates[4])
        portfolio.sell("INFY", 1470, 50, dates[5])
        return portfolio

    def test_sharpe_ratio_exact_match(self):
        portfolio = self._driven_portfolio()
        stats = portfolio.get_stats()
        equity = pd.Series([entry["equity"] for entry in portfolio.equity_curve])
        returns = equity.pct_change().dropna()
        expected = empyrical.sharpe_ratio(returns, risk_free=_daily_rf())
        # get_stats rounds Sharpe to 2 decimals
        assert stats["sharpe_ratio"] == pytest.approx(expected, abs=6e-3)

    def test_max_drawdown_matches_empyrical(self):
        portfolio = self._driven_portfolio()
        stats = portfolio.get_stats()
        equity = pd.Series([entry["equity"] for entry in portfolio.equity_curve])
        returns = equity.pct_change().dropna()
        expected = empyrical.max_drawdown(returns) * 100.0
        # get_stats rounds max_drawdown to 1 decimal (%)
        assert stats["max_drawdown"] == pytest.approx(expected, abs=0.06)


class TestComputeMetricsGuardrails:
    """Behaviour we keep while the library comparison runs on valid paths."""

    def test_single_point_equity_curve_returns_empty(self):
        from src.trading.backtester import compute_metrics

        frame = pd.DataFrame({"date": [pd.Timestamp("2024-06-01").date()], "equity": [100.0]})
        assert compute_metrics(frame, [], risk_free_rate=_risk_free_rate()) == {}

    def test_zero_start_equity_returns_empty(self):
        from src.trading.backtester import compute_metrics

        frame, _ = _equity_frame(seed=1, n=50)
        frame.loc[0, "equity"] = 0.0
        assert compute_metrics(frame, [], risk_free_rate=_risk_free_rate()) == {}

    def test_zero_volatility_series_returns_zero_periodic_ratios(self):
        # Constant equity: returns are exactly 0, std is exactly 0 ->
        # Sharpe/Calmar must guard to 0 instead of NaN/inf leaking into
        # downstream gates. Sortino is intentionally NOT zero: zero return
        # sits below the risk-free MAR, so the ratio is exactly -sqrt(252)
        # (excess shortfall rf over rf-driven downside deviation). Empyrical
        # is not a comparator here (its Sortino numerator is arithmetic-mean
        # based), so we assert the mathematical invariant directly. (A
        # constant-percentage-growth path is NOT zero-vol in floating
        # point: 1.001**k leaves ULP dust so std > 0, producing an
        # astronomically large ratio — degenerate-input behaviour outside
        # this test's scope.)
        from src.trading.backtester import compute_metrics

        n = 260
        equity = np.full(n, 100.0)
        frame = pd.DataFrame({"date": list(pd.bdate_range("2023-01-02", periods=n)), "equity": equity})
        metrics = compute_metrics(frame, [], risk_free_rate=_risk_free_rate())
        assert metrics["sharpe_ratio"] == 0.0
        assert metrics["sortino_ratio"] == pytest.approx(-np.sqrt(252), abs=6e-4)
        assert metrics["calmar_ratio"] == 0.0
        assert metrics["annualized_volatility"] == 0.0
        assert metrics["max_drawdown"] == 0.0
        assert metrics["var_95"] == 0.0
        assert metrics["cvar_95"] == 0.0

    def test_trade_win_rate_and_profit_factor_internal_consistency(self):
        from src.trading.backtester import compute_metrics

        frame, _ = _equity_frame(seed=5, n=60)
        pnls = [500.0, 200.0, -200.0, -50.0]
        metrics = compute_metrics(frame, _trades_list(pnls), risk_free_rate=_risk_free_rate())
        assert metrics["total_trades"] == 4
        assert metrics["win_rate"] == 50.0  # 2 of 4 wins
        # avg_win 350, avg_loss 125 -> profit factor 700/250 == 2.8
        assert metrics["avg_win"] == 350.0
        assert metrics["avg_loss"] == 125.0
        assert metrics["profit_factor"] == 2.8

    def test_all_metrics_finite_across_random_paths(self):
        from src.trading.backtester import compute_metrics

        for seed, mu, sigma in TestConventionIdenticalParity.SEEDS:
            frame, _ = _equity_frame(seed=seed, mu=mu, sigma=sigma)
            metrics = compute_metrics(frame, _trades_list([120.0, -80.0, 40.0]), risk_free_rate=_risk_free_rate())
            assert metrics, "metrics dict unexpectedly empty"
            for key, value in metrics.items():
                numeric = not isinstance(value, (str, int))
                if numeric:
                    assert np.isfinite(value), f"non-finite {key} for seed={seed}: {value}"

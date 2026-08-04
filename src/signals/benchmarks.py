"""Benchmark strategies for comparison.

Provides baseline strategies to test whether a model adds value:
- Buy and Hold
- 20-day Momentum
- Always Long
- Random
- Simple Moving Average Crossover
"""
import numpy as np
import pandas as pd
from src.core.constants import RISK_FREE_RATE


def buy_and_hold(actuals, returns):
    """Benchmark: buy and hold throughout the test period.

    Args:
        actuals: Not used (for interface consistency)
        returns: Daily returns

    Returns:
        Dict with cumulative return, Sharpe, accuracy (always predicts UP)
    """
    returns = np.asarray(returns)
    cum_return = float(np.prod(1 + returns) - 1)

    if returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((returns.mean() - daily_rf) / returns.std() * np.sqrt(252))

    return {
        "name": "Buy & Hold",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": 1.0,  # Always "in the market"
        "n_trades": 1,
        "annualized_return": float((1 + cum_return) ** (252 / max(len(returns), 1)) - 1),
    }


def momentum_20d(returns):
    """Benchmark: go long when 20-day momentum is positive.

    Args:
        returns: Daily returns

    Returns:
        Dict with performance metrics
    """
    returns = np.asarray(returns)
    signals = np.zeros(len(returns))

    for i in range(20, len(returns)):
        past_returns = returns[i-20:i]
        cum_ret = np.prod(1 + past_returns) - 1
        signals[i] = 1 if cum_ret > 0 else 0

    strategy_returns = signals * returns
    cum_return = float(np.prod(1 + strategy_returns) - 1)

    if strategy_returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((strategy_returns.mean() - daily_rf) /
                       strategy_returns.std() * np.sqrt(252))

    n_trades = int(np.sum(np.abs(np.diff(signals))))

    return {
        "name": "20D Momentum",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": float(np.mean(signals[20:] == (returns[20:] > 0).astype(int))),
        "n_trades": n_trades,
        "annualized_return": float((1 + cum_return) ** (252 / max(len(returns), 1)) - 1),
    }


def always_long(returns):
    """Benchmark: always predict UP.

    Args:
        returns: Daily returns

    Returns:
        Dict with performance metrics
    """
    returns = np.asarray(returns)
    cum_return = float(np.prod(1 + returns) - 1)

    if returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((returns.mean() - daily_rf) / returns.std() * np.sqrt(252))

    accuracy = float(np.mean(returns > 0))

    return {
        "name": "Always Long",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": accuracy,
        "n_trades": 0,
        "annualized_return": float((1 + cum_return) ** (252 / max(len(returns), 1)) - 1),
    }


def random_strategy(returns, random_state=42):
    """Benchmark: randomly go long or flat.

    Args:
        returns: Daily returns
        random_state: Random seed

    Returns:
        Dict with performance metrics
    """
    returns = np.asarray(returns)
    rng = np.random.RandomState(random_state)
    signals = rng.randint(0, 2, len(returns))

    strategy_returns = signals * returns
    cum_return = float(np.prod(1 + strategy_returns) - 1)

    if strategy_returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((strategy_returns.mean() - daily_rf) /
                       strategy_returns.std() * np.sqrt(252))

    return {
        "name": "Random",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": float(np.mean(signals == (returns > 0).astype(int))),
        "n_trades": int(np.sum(np.abs(np.diff(signals)))),
        "annualized_return": float((1 + cum_return) ** (252 / max(len(returns), 1)) - 1),
    }


def sma_crossover(returns, short_window=10, long_window=50):
    """Benchmark: SMA crossover (short > long = long).

    Args:
        returns: Daily returns
        short_window: Short SMA window
        long_window: Long SMA window

    Returns:
        Dict with performance metrics
    """
    returns = np.asarray(returns)
    price_series = np.cumprod(1 + returns)
    sma_short = pd.Series(price_series).rolling(short_window).mean().values
    sma_long = pd.Series(price_series).rolling(long_window).mean().values

    # Use NaN for warmup periods (no signal = no position = flat)
    signals = np.where(np.isnan(sma_short) | np.isnan(sma_long), 0,
                       np.where(sma_short > sma_long, 1, 0))
    strategy_returns = signals * returns
    cum_return = float(np.prod(1 + strategy_returns) - 1)

    if strategy_returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((strategy_returns.mean() - daily_rf) /
                       strategy_returns.std() * np.sqrt(252))

    return {
        "name": f"SMA {short_window}/{long_window}",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": float(np.mean(signals == (returns > 0).astype(int))),
        "n_trades": int(np.sum(np.abs(np.diff(signals)))),
        "annualized_return": float((1 + cum_return) ** (252 / max(len(returns), 1)) - 1),
    }


def compare_to_benchmarks(strategy_returns, actuals=None, returns=None):
    """Compare strategy to all benchmarks.

    Args:
        strategy_returns: Array of strategy daily returns
        actuals: Array of actual directions (0/1)
        returns: Array of daily returns (if None, derived from strategy_returns)

    Returns:
        List of benchmark dicts
    """
    strategy_returns = np.asarray(strategy_returns)

    if returns is None:
        # Use strategy returns as the best available proxy
        returns = strategy_returns

    benchmarks = [
        buy_and_hold(actuals, returns),
        momentum_20d(returns),
        always_long(returns),
        random_strategy(returns),
        sma_crossover(returns),
    ]

    # Add strategy performance
    cum_return = float(np.prod(1 + strategy_returns) - 1)
    if strategy_returns.std() == 0:
        sharpe = 0.0
    else:
        daily_rf = RISK_FREE_RATE / 252
        sharpe = float((strategy_returns.mean() - daily_rf) /
                       strategy_returns.std() * np.sqrt(252))

    accuracy = float(np.mean(
        (strategy_returns[1:] > 0) == (np.asarray(returns)[1:] > 0)
    )) if len(strategy_returns) > 1 else 0

    strategy_result = {
        "name": "Your Strategy",
        "cumulative_return": cum_return,
        "sharpe": sharpe,
        "accuracy": accuracy,
        "n_trades": int(np.sum(np.abs(np.diff((strategy_returns > 0).astype(int))))),
        "annualized_return": float((1 + cum_return) ** (252 / max(len(strategy_returns), 1)) - 1),
    }

    all_results = [strategy_result] + benchmarks
    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    return all_results


def _max_drawdown_from_curve(values: list[float]) -> float:
    """Max drawdown from an equity curve (peak to trough)."""
    peak = 0.0
    max_dd = 0.0
    for v in values:
        if v > peak:
            peak = v
        if peak > 0:
            max_dd = max(max_dd, (peak - v) / peak)
    return max_dd


def paper_vs_nifty(ledger) -> dict:
    """Compare the paper trading equity curve against Nifty 50 buy-and-hold.

    This is the single most honest signal of whether the system adds value:
    does the paper portfolio beat doing nothing but holding the index, net
    of the paper trading costs already modeled in the snapshots?

    Returns:
        Dict with per-date curves, cumulative returns, alpha, information
        ratio, and max drawdowns — or an empty "error" dict when there is
        insufficient evidence.
    """
    snapshots = ledger.get_snapshots()
    if not snapshots:
        return {"error": "no portfolio snapshots yet"}

    snapshots = sorted(snapshots, key=lambda s: str(s["date"]))
    start_date = str(snapshots[0]["date"])[:10]
    end_date = str(snapshots[-1]["date"])[:10]
    if start_date >= end_date:
        return {"error": "need at least 2 snapshot dates for a benchmark comparison"}

    # Paper equity curve
    paper_dates = [str(s["date"])[:10] for s in snapshots]
    paper_values = [float(s["total_value"]) for s in snapshots]
    paper_cum = [v / paper_values[0] - 1.0 for v in paper_values]

    # Nifty 50 over the same window (adjusted close)
    try:
        from src.data.data_fetcher import fetch_stock_data
        nifty = fetch_stock_data("^NSEI", period="2y", force_refresh=False)
    except Exception as e:
        return {"error": f"benchmark data unavailable: {e}"}
    if nifty is None or len(nifty) < 2:
        return {"error": "benchmark data unavailable"}

    nifty_closes = nifty["close"].astype(float)
    nifty_closes.index = pd.to_datetime(nifty_closes.index)
    window = nifty_closes[(nifty_closes.index >= pd.Timestamp(start_date))
                          & (nifty_closes.index <= pd.Timestamp(end_date))]
    if len(window) < 2:
        return {"error": "benchmark window too short"}

    # Nifty buy-and-hold: first close -> last close (no dividends modeled)
    nifty_cum = window.values / float(window.values[0]) - 1.0

    # Rebase paper curve to Nifty dates for a like-for-like chart: keep both
    # as independent series keyed by their own dates (front-end aligns).
    nifty_dates = [d.strftime("%Y-%m-%d") for d in window.index]

    # Alpha: end-of-period difference in cumulative return
    nifty_total = float(window.values[-1] / window.values[0] - 1.0)
    paper_total = paper_cum[-1]
    alpha = paper_total - nifty_total

    # Information ratio: annualized mean/std of daily return differences
    paper_returns = np.diff(paper_values) / np.asarray(paper_values[:-1])
    nifty_returns = np.diff(window.values) / window.values[:-1]
    if len(paper_returns) > 1 and len(nifty_returns) > 1:
        n = min(len(paper_returns), len(nifty_returns))
        diff = paper_returns[-n:] - nifty_returns[-n:]
        ir = float(diff.mean() / diff.std() * np.sqrt(252)) if diff.std() > 0 else 0.0
    else:
        ir = None

    return {
        "start_date": start_date,
        "end_date": end_date,
        "paper": {"dates": paper_dates, "cumulative": [round(c, 6) for c in paper_cum]},
        "nifty": {"dates": nifty_dates, "cumulative": [round(c, 6) for c in nifty_cum.tolist()]},
        "paper_total_return": round(paper_total, 6),
        "nifty_total_return": round(nifty_total, 6),
        "alpha": round(alpha, 6),
        "information_ratio": round(ir, 4) if ir is not None else None,
        "paper_max_drawdown": round(_max_drawdown_from_curve(paper_cum), 4),
        "nifty_max_drawdown": round(_max_drawdown_from_curve(nifty_cum.tolist()), 4),
        "n_snapshots": len(snapshots),
        "nifty_points": len(window),
    }

"""Backtest scenarios module.

Allows users to test different strategies:
- Different feature sets
- Different models
- Different timeframes
- Different risk parameters
"""
import numpy as np
from src.core.constants import RISK_FREE_RATE


def _max_drawdown(returns):
    """Compute max drawdown from returns array."""
    if len(returns) == 0:
        return 0.0
    equity = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(running_max > 0, (equity - running_max) / running_max, 0.0)
    return float(np.nanmin(drawdown))


def scenario_buy_and_hold(df):
    """Simple buy and hold baseline."""
    close = df["close"]
    returns = close.pct_change().dropna()

    total_return = float(close.iloc[-1] / close.iloc[0] - 1) if close.iloc[0] > 0 else 0.0
    n_years = max(len(returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1) if total_return > -1 else -1.0
    ann_vol = float(returns.std() * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    return {
        "name": "Buy & Hold",
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(returns.values),
        "n_trades": 1,
    }


def scenario_momentum(df, lookback=20):
    """Simple momentum strategy."""
    close = df["close"]
    returns = close.pct_change()

    signals = []
    for i in range(lookback, len(close)):
        past_return = close.iloc[i] / close.iloc[i - lookback] - 1
        signals.append(1 if past_return > 0 else 0)

    signals = np.array(signals)
    test_returns = returns.iloc[lookback:].values
    strategy_returns = signals * test_returns

    total_return = float(np.prod(1 + strategy_returns) - 1)
    n_years = max(len(strategy_returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    ann_vol = float(np.std(strategy_returns) * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    n_trades = int(np.sum(np.abs(np.diff(signals))))

    return {
        "name": f"Momentum ({lookback}d)",
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(strategy_returns),
        "n_trades": n_trades,
    }


def scenario_mean_reversion(df, window=20):
    """Mean reversion strategy."""
    close = df["close"]
    returns = close.pct_change()

    signals = []
    for i in range(window, len(close)):
        sma = close.iloc[i - window:i].mean()
        std = close.iloc[i - window:i].std()
        if close.iloc[i] < sma - std:
            signals.append(1)  # Oversold, buy
        elif close.iloc[i] > sma + std:
            signals.append(0)  # Overbought, sell
        else:
            signals.append(0)

    signals = np.array(signals)
    test_returns = returns.iloc[window:].values
    strategy_returns = signals * test_returns

    total_return = float(np.prod(1 + strategy_returns) - 1)
    n_years = max(len(strategy_returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    ann_vol = float(np.std(strategy_returns) * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    n_trades = int(np.sum(np.abs(np.diff(signals))))

    return {
        "name": f"Mean Reversion ({window}d)",
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(strategy_returns),
        "n_trades": n_trades,
    }


def scenario_volatility_target(df, target_vol=0.15, window=20):
    """Volatility-targeting strategy."""
    close = df["close"]
    returns = close.pct_change().dropna()

    # Compute rolling vol
    rolling_vol = returns.rolling(window).std() * np.sqrt(252)
    rolling_vol = rolling_vol.fillna(0.20)

    # Target vol position sizing
    positions = np.minimum(target_vol / rolling_vol.values, 1.0)
    positions = np.maximum(positions, 0.0)

    strategy_returns = positions * returns.values

    total_return = float(np.prod(1 + strategy_returns) - 1)
    n_years = max(len(strategy_returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    ann_vol = float(np.std(strategy_returns) * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    return {
        "name": f"Vol Target ({target_vol:.0%})",
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(strategy_returns),
        "n_trades": 0,
    }


def scenario_sma_crossover(df, short_window=10, long_window=50):
    """SMA crossover strategy."""
    close = df["close"]
    returns = close.pct_change()

    sma_short = close.rolling(short_window).mean()
    sma_long = close.rolling(long_window).mean()

    signals = []
    for i in range(long_window, len(close)):
        if sma_short.iloc[i] > sma_long.iloc[i]:
            signals.append(1)
        else:
            signals.append(0)

    signals = np.array(signals)
    test_returns = returns.iloc[long_window:].values
    strategy_returns = signals * test_returns

    total_return = float(np.prod(1 + strategy_returns) - 1)
    n_years = max(len(strategy_returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    ann_vol = float(np.std(strategy_returns) * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0
    n_trades = int(np.sum(np.abs(np.diff(signals))))

    return {
        "name": f"SMA {short_window}/{long_window}",
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": float(sharpe),
        "max_drawdown": _max_drawdown(strategy_returns),
        "n_trades": n_trades,
    }


def run_all_scenarios(df):
    """Run all backtest scenarios on a stock.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        List of scenario results sorted by Sharpe
    """
    scenarios = [
        scenario_buy_and_hold(df),
        scenario_momentum(df, lookback=10),
        scenario_momentum(df, lookback=20),
        scenario_momentum(df, lookback=60),
        scenario_mean_reversion(df, window=10),
        scenario_mean_reversion(df, window=20),
        scenario_volatility_target(df, target_vol=0.10),
        scenario_volatility_target(df, target_vol=0.15),
        scenario_volatility_target(df, target_vol=0.20),
        scenario_sma_crossover(df, short_window=10, long_window=50),
        scenario_sma_crossover(df, short_window=20, long_window=60),
    ]

    scenarios.sort(key=lambda x: x["sharpe"], reverse=True)
    return scenarios

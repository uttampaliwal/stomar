"""Regime-conditional strategy module.

Provides rules-based trading that adapts to market conditions:
- Bull regime: Long only, aggressive
- Bear regime: Defensive, reduce exposure
- Sideways: Mean reversion, range-bound
"""
import numpy as np
from src.regime import detect_regime


def get_regime_allocation(regime, confidence=0.5):
    """Get recommended allocation based on regime.

    Args:
        regime: "Bull", "Bear", or "Sideways"
        confidence: Regime detection confidence (0-1)

    Returns:
        Dict with equity_pct, cash_pct, strategy, reasoning
    """
    base_allocations = {
        "Bull": {
            "equity_pct": 0.80,
            "cash_pct": 0.20,
            "strategy": "momentum",
            "reasoning": "Bull market — ride the trend, stay mostly invested",
        },
        "Sideways": {
            "equity_pct": 0.50,
            "cash_pct": 0.50,
            "strategy": "mean_reversion",
            "reasoning": "Sideways market — range-bound, take profits at resistance",
        },
        "Bear": {
            "equity_pct": 0.20,
            "cash_pct": 0.80,
            "strategy": "defensive",
            "reasoning": "Bear market — preserve capital, only high-conviction trades",
        },
    }

    alloc = base_allocations.get(regime, base_allocations["Sideways"])

    # Adjust by confidence
    if confidence < 0.3:
        alloc["reasoning"] += " (low confidence — be more conservative)"
        alloc["equity_pct"] *= 0.7
        alloc["cash_pct"] = 1 - alloc["equity_pct"]

    return alloc


def regime_adjusted_position_size(base_position_pct, regime, current_vol=None):
    """Adjust position size based on regime.

    Args:
        base_position_pct: Base position size (fraction of portfolio)
        regime: Current regime
        current_vol: Current annualized volatility

    Returns:
        Dict with adjusted position size and reasoning
    """
    regime_multipliers = {
        "Bull": 1.2,
        "Sideways": 1.0,
        "Bear": 0.5,
    }

    multiplier = regime_multipliers.get(regime, 1.0)
    adjusted = base_position_pct * multiplier

    # Additional vol adjustment
    if current_vol is not None and current_vol > 0:
        vol_scalar = 0.20 / current_vol  # Target 20% vol
        vol_scalar = max(0.5, min(2.0, vol_scalar))
        adjusted *= vol_scalar

    adjusted = min(adjusted, 0.25)  # Max 25% per position

    return {
        "position_size_pct": float(adjusted),
        "regime_multiplier": float(multiplier),
        "reasoning": f"Regime: {regime} — {multiplier:.1f}x base size",
    }


def generate_regime_signals(df, regime_result=None):
    """Generate trading signals based on regime.

    Args:
        df: DataFrame with OHLCV data
        regime_result: Pre-computed regime detection result

    Returns:
        Dict with signals and regime info
    """
    close = df["close"]
    returns = close.pct_change().dropna()

    if regime_result is None:
        regime_result = detect_regime(close, ohlc=df)

    regime = regime_result["regime"]
    confidence = regime_result["confidence"]

    allocation = get_regime_allocation(regime, confidence)

    # Generate simple signals based on regime
    signals = []
    for i in range(1, len(df)):
        price = df.iloc[i]["close"]
        prev_price = df.iloc[i - 1]["close"]

        if regime == "Bull":
            signal = 1 if price > prev_price else 0
        elif regime == "Bear":
            signal = 0  # Stay out
        else:  # Sideways
            sma20 = close.iloc[max(0, i - 20):i].mean()
            signal = 1 if price < sma20 else 0  # Buy below SMA

        signals.append({
            "date": str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i]),
            "signal": signal,
            "price": float(price),
        })

    return {
        "regime": regime,
        "confidence": confidence,
        "allocation": allocation,
        "signals": signals,
    }


def backtest_regime_strategy(df, regime_result=None):
    """Backtest regime-based strategy.

    Args:
        df: DataFrame with OHLCV data
        regime_result: Pre-computed regime detection result

    Returns:
        Dict with backtest results
    """
    close = df["close"]
    returns = close.pct_change().dropna()

    if regime_result is None:
        regime_result = detect_regime(close, ohlc=df)

    regime = regime_result["regime"]
    allocation = get_regime_allocation(regime, regime_result["confidence"])

    equity_pct = allocation["equity_pct"]

    # Simple backtest: hold equity_pct in stocks, rest in cash
    daily_returns = returns.values
    strategy_returns = daily_returns * equity_pct

    cumulative = np.cumprod(1 + strategy_returns)
    total_return = float(cumulative[-1] - 1) if len(cumulative) > 0 else 0

    n_years = max(len(returns) / 252, 0.01)
    ann_return = float((1 + total_return) ** (1 / n_years) - 1)
    ann_vol = float(np.std(strategy_returns) * np.sqrt(252))
    sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else 0

    # Buy and hold comparison
    bh_return = float(close.iloc[-1] / close.iloc[0] - 1) if len(close) > 1 else 0
    bh_ann = float((1 + bh_return) ** (1 / n_years) - 1)

    return {
        "regime": regime,
        "equity_allocation": equity_pct,
        "strategy_return": total_return,
        "strategy_annualized": ann_return,
        "strategy_sharpe": float(sharpe),
        "buy_hold_return": bh_return,
        "buy_hold_annualized": bh_ann,
        "excess_return": float(ann_return - bh_ann),
        "n_days": len(returns),
    }

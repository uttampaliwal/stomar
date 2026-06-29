"""Volatility forecasting module.

Provides multiple volatility estimation and forecasting methods:
- Historical volatility (rolling window)
- Exponentially Weighted Moving Average (EWMA)
- Parkinson volatility (high-low range)
- Garman-Klass volatility (OHLC)
- Yang-Zhang volatility (OHLC, unbiased)
- Simple GARCH-like forecasting
- Volatility regime detection
"""
import numpy as np
import pandas as pd
from scipy import stats


def historical_volatility(returns, window=20, annualize=True):
    """Compute historical volatility using rolling standard deviation.

    Args:
        returns: Series of daily returns
        window: Rolling window size
        annualize: If True, multiply by sqrt(252)

    Returns:
        Series of volatility values
    """
    vol = returns.rolling(window).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def ewma_volatility(returns, span=20, annualize=True):
    """Compute EWMA volatility (more weight on recent data).

    Args:
        returns: Series of daily returns
        span: EWMA span
        annualize: If True, multiply by sqrt(252)

    Returns:
        Series of volatility values
    """
    vol = returns.ewm(span=span).std()
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def parkinson_volatility(high, low, window=20, annualize=True):
    """Parkinson volatility estimator using high-low range.

    More efficient than close-to-close because it uses intraday range.

    Args:
        high: Series of high prices
        low: Series of low prices
        window: Rolling window size
        annualize: If True, multiply by sqrt(252)

    Returns:
        Series of volatility values
    """
    log_hl = np.log(high / low)
    vol = np.sqrt((log_hl ** 2).rolling(window).mean() / (4 * np.log(2)))
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def garman_klass_volatility(open_, high, low, close, window=20, annualize=True):
    """Garman-Klass volatility estimator using OHLC.

    More efficient than Parkinson because it uses open and close too.

    Args:
        open_: Series of open prices
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: Rolling window size
        annualize: If True, multiply by sqrt(252)

    Returns:
        Series of volatility values
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    vol = np.sqrt(
        (0.5 * log_hl ** 2 - (2 * np.log(2) - 1) * log_co ** 2).rolling(window).mean()
    )
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def yang_zhang_volatility(open_, high, low, close, window=20, annualize=True):
    """Yang-Zhang volatility estimator (unbiased for opening jumps).

    Best OHLC estimator when there are overnight gaps.

    Args:
        open_: Series of open prices
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: Rolling window (must be > 1)
        annualize: If True, multiply by sqrt(252)

    Returns:
        Series of volatility values
    """
    if window <= 1:
        window = 2

    log_co = np.log(close / open_)
    log_oc = np.log(open_ / close.shift(1))
    log_ho = np.log(high / open_)
    log_lo = np.log(low / open_)

    k = 0.34 / (1.34 + (window + 1) / (window - 1))
    o_var = log_oc.rolling(window).var()
    c_var = log_co.rolling(window).var()
    rs_var = (log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)).rolling(window).mean()

    vol = np.sqrt(o_var + k * c_var + (1 - k) * rs_var)
    if annualize:
        vol = vol * np.sqrt(252)
    return vol


def forecast_volatility(returns, method="ewma", horizon=5, span=20):
    """Forecast future volatility.

    Args:
        returns: Series of daily returns
        method: "ewma" or "historical"
        horizon: Days to forecast
        span: EWMA span (if method="ewma")

    Returns:
        Dict with forecast_vol, confidence_intervals
    """
    if not isinstance(returns, pd.Series):
        returns = pd.Series(returns)
    returns = returns.dropna()

    if len(returns) < span:
        return {
            "current_vol": 0.0, "forecast_vols": [0.0] * horizon,
            "long_term_vol": 0.0, "vol_std": 0.0,
            "horizon": horizon, "method": method,
        }

    if method == "ewma":
        current_vol = returns.ewm(span=span).std().iloc[-1]
    else:
        current_vol = returns.rolling(span).std().iloc[-1]

    ann_vol = current_vol * np.sqrt(252)

    # Simple forecast: assume vol reverts to mean
    long_term_vol = returns.std() * np.sqrt(252)
    mean_reversion = 0.1  # 10% mean reversion per day

    forecast_vols = []
    for h in range(1, horizon + 1):
        f = ann_vol * (1 - mean_reversion) ** h + long_term_vol * (1 - (1 - mean_reversion) ** h)
        forecast_vols.append(f)

    # Confidence intervals (assuming normal distribution of vol)
    vol_std = returns.rolling(span).std().iloc[-1] * np.sqrt(252) * 0.2

    return {
        "current_vol": float(ann_vol),
        "forecast_vols": forecast_vols,
        "long_term_vol": float(long_term_vol),
        "vol_std": float(vol_std),
        "horizon": horizon,
        "method": method,
    }


def detect_volatility_regime(returns, window=20, threshold_low=0.15, threshold_high=0.25):
    """Detect volatility regime (Low, Medium, High).

    Args:
        returns: Series of daily returns
        window: Rolling window for vol calculation
        threshold_low: Below this = Low vol regime
        threshold_high: Above this = High vol regime

    Returns:
        Dict with current regime, vol percentile, and regime history
    """
    ann_vol = historical_volatility(returns, window=window, annualize=True)
    ann_vol = ann_vol.dropna()

    if len(ann_vol) == 0:
        return {
            "current_regime": "Medium",
            "current_vol": 0.0,
            "vol_percentile": 50.0,
            "regime_color": "orange",
            "regime_percentages": {"Low": 0.33, "Medium": 0.34, "High": 0.33},
            "threshold_low": threshold_low,
            "threshold_high": threshold_high,
        }

    current_vol = ann_vol.iloc[-1]

    # Percentile of current vol in history
    vol_percentile = float(stats.percentileofscore(ann_vol.dropna(), current_vol))

    if current_vol < threshold_low:
        regime = "Low"
        color = "green"
    elif current_vol > threshold_high:
        regime = "High"
        color = "red"
    else:
        regime = "Medium"
        color = "orange"

    # Regime history
    regimes = pd.Series(index=ann_vol.index, dtype=str)
    regimes[ann_vol < threshold_low] = "Low"
    regimes[ann_vol > threshold_high] = "High"
    regimes[(ann_vol >= threshold_low) & (ann_vol <= threshold_high)] = "Medium"

    # Time in each regime (last 252 days)
    recent = regimes.iloc[-252:]
    regime_pct = {
        "Low": float((recent == "Low").mean()),
        "Medium": float((recent == "Medium").mean()),
        "High": float((recent == "High").mean()),
    }

    return {
        "current_regime": regime,
        "current_vol": float(current_vol),
        "vol_percentile": vol_percentile,
        "regime_color": color,
        "regime_percentages": regime_pct,
        "threshold_low": threshold_low,
        "threshold_high": threshold_high,
    }


def compute_bollinger_bands(close, window=20, num_std=2):
    """Compute Bollinger Bands.

    Args:
        close: Series of close prices
        window: Rolling window
        num_std: Number of standard deviations

    Returns:
        Dict with upper, middle, lower bands and bandwidth
    """
    middle = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    bandwidth = (upper - lower) / middle

    return {
        "upper": upper,
        "middle": middle,
        "lower": lower,
        "bandwidth": bandwidth,
        "percent_b": (close - lower) / (upper - lower),
    }


def volatility_cone(high, low, close, windows=[5, 10, 20, 60, 120]):
    """Compute volatility cone (min, max, avg vol at different windows).

    Useful for understanding current vol relative to historical ranges.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        windows: List of window sizes

    Returns:
        Dict with min, max, mean, current vol for each window
    """
    returns = close.pct_change()
    cone = {}

    for w in windows:
        hist_vol = historical_volatility(returns, window=w, annualize=True)
        cone[w] = {
            "min": float(hist_vol.min()),
            "max": float(hist_vol.max()),
            "mean": float(hist_vol.mean()),
            "current": float(hist_vol.iloc[-1]),
            "percentile": float(stats.percentileofscore(hist_vol.dropna(), hist_vol.iloc[-1])),
        }

    return cone


def compute_atr(high, low, close, window=14):
    """Compute Average True Range.

    Args:
        high: Series of high prices
        low: Series of low prices
        close: Series of close prices
        window: ATR window

    Returns:
        Series of ATR values
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    return atr


def position_size_for_vol(risk_per_trade=0.02, stop_loss_pct=0.05,
                          current_vol=None, target_vol=0.15):
    """Compute position size adjusted for volatility.

    Uses vol-targeting: when vol is high, reduce position; when low, increase.

    Args:
        risk_per_trade: Maximum risk per trade (fraction of capital)
        stop_loss_pct: Stop loss percentage
        current_vol: Current annualized volatility
        target_vol: Target annualized volatility

    Returns:
        Dict with position_size_pct, vol_scalar, reasoning
    """
    if current_vol is None or current_vol <= 0:
        return {
            "position_size_pct": risk_per_trade / stop_loss_pct,
            "vol_scalar": 1.0,
            "reasoning": "No vol data, using default sizing",
        }

    vol_scalar = target_vol / current_vol
    vol_scalar = max(0.5, min(2.0, vol_scalar))  # Cap between 0.5x and 2x

    base_size = risk_per_trade / stop_loss_pct
    adjusted_size = base_size * vol_scalar

    if current_vol < 0.15:
        reasoning = f"Low vol ({current_vol:.1%}) — can increase position"
    elif current_vol > 0.25:
        reasoning = f"High vol ({current_vol:.1%}) — reduce position"
    else:
        reasoning = f"Normal vol ({current_vol:.1%}) — standard sizing"

    return {
        "position_size_pct": float(min(adjusted_size, 0.5)),
        "vol_scalar": float(vol_scalar),
        "base_size": float(base_size),
        "reasoning": reasoning,
    }


def full_volatility_analysis(df, window=20):
    """Run complete volatility analysis on a stock.

    Args:
        df: DataFrame with OHLCV data
        window: Default window for vol calculations

    Returns:
        Dict with all volatility metrics
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    returns = close.pct_change().dropna()

    # Multiple volatility estimators
    hist_vol = historical_volatility(returns, window=window)
    ewma_vol = ewma_volatility(returns, span=window)
    park_vol = parkinson_volatility(high, low, window=window)
    gk_vol = garman_klass_volatility(open_, high, low, close, window=window)
    yz_vol = yang_zhang_volatility(open_, high, low, close, window=window)

    # Volatility regime
    regime = detect_volatility_regime(returns, window=window)

    # Forecast
    forecast = forecast_volatility(returns, method="ewma", horizon=5)

    # Bollinger Bands
    bollinger = compute_bollinger_bands(close, window=window)

    # ATR
    atr = compute_atr(high, low, close, window=14)

    # Volatility cone
    cone = volatility_cone(high, low, close)

    # Position sizing
    pos_size = position_size_for_vol(current_vol=regime["current_vol"])

    # Current values
    current = {
        "hist_vol": float(hist_vol.iloc[-1]) if len(hist_vol) > 0 else 0,
        "ewma_vol": float(ewma_vol.iloc[-1]) if len(ewma_vol) > 0 else 0,
        "parkinson_vol": float(park_vol.iloc[-1]) if len(park_vol) > 0 else 0,
        "garman_klass_vol": float(gk_vol.iloc[-1]) if len(gk_vol) > 0 else 0,
        "yang_zhang_vol": float(yz_vol.iloc[-1]) if len(yz_vol) > 0 else 0,
        "atr": float(atr.iloc[-1]) if len(atr) > 0 else 0,
        "atr_pct": float(atr.iloc[-1] / close.iloc[-1]) if len(atr) > 0 and close.iloc[-1] > 0 else 0,
        "bollinger_bandwidth": float(bollinger["bandwidth"].iloc[-1]) if len(bollinger["bandwidth"]) > 0 else 0,
        "bollinger_pct_b": float(bollinger["percent_b"].iloc[-1]) if len(bollinger["percent_b"]) > 0 else 0,
    }

    return {
        "current": current,
        "regime": regime,
        "forecast": forecast,
        "bollinger": bollinger,
        "cone": cone,
        "position_sizing": pos_size,
        "estimators": {
            "historical": hist_vol,
            "ewma": ewma_vol,
            "parkinson": park_vol,
            "garman_klass": gk_vol,
            "yang_zhang": yz_vol,
        },
    }

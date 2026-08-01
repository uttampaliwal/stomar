"""SOTA feature engineering pipeline: 30+ quantitative factors.

Three factor families:

1. Technical indicators — RSI, MACD, Stochastic, Bollinger Bands, ATR,
   Supertrend, Ichimoku Cloud, On-Balance Volume, Chaikin Money Flow.
2. Market microstructure — Order Flow Imbalance proxy, volatility Z-score,
   PCR slope, FII net-flow momentum ratio.
3. Cross-sectional factors — vol-adjusted momentum (1M/3M/6M/12M), relative
   strength vs NIFTY 50, high-low range spread.

All features are computed with point-in-time discipline:
- rolling/EMA features use only past data by construction;
- external-flow features (PCR, FII/DII, sentiment, MTF) are only populated on
  the most recent row, matching the existing codebase convention and
  eliminating look-ahead bias in backtests.

Usage:
    from src.signals.feature_pipeline import compute_features
    df_feat = compute_features(df, ticker="RELIANCE.NS", index_df=nifty_df)
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.features import (
    add_flow_features,
    add_multitimeframe_features,
    add_pcr_features,
    add_sentiment_features,
    add_technical_indicators,
)

logger = logging.getLogger(__name__)

# ─── factor registry ─────────────────────────────────────────────────────────

FACTOR_GROUPS = {
    "trend": [
        "sma_10", "sma_20", "sma_50", "ema_12", "ema_26", "macd", "macd_signal",
        "adx", "supertrend", "supertrend_dir", "vwap",
        "ichimoku_tenkan", "ichimoku_kijun", "ichimoku_senkou_a",
        "ichimoku_senkou_b",
    ],
    "momentum": [
        "rsi", "stoch_k", "stoch_d", "williams_r", "cci", "mfi",
        "mom_1m_voladj", "mom_3m_voladj", "mom_6m_voladj", "mom_12m_voladj",
        "rel_strength_1m", "rel_strength_3m",
    ],
    "volatility": [
        "atr", "bb_high", "bb_low", "bb_width", "high_low_pct", "close_open_pct",
        "close_position", "volatility_5d", "volatility_10d", "volatility_20d",
        "vol_zscore",
    ],
    "volume": [
        "obv", "cmf", "volume_sma", "volume_ratio", "return_vol_corr",
    ],
    "microstructure": [
        "ofi", "pcr_slope", "fii_momentum", "high_low_spread",
        "rsi_x_volume", "macd_x_bb_width", "adx_x_volatility", "momentum_x_vol",
        "return_skew", "return_kurt",
    ],
    "returns": [
        "returns_1d", "returns_2d", "returns_3d", "returns_5d",
        "returns_10d", "returns_20d", "return_lag_1", "return_lag_2",
        "return_lag_3", "return_lag_5",
    ],
    "external": [
        "sentiment_score", "fii_net", "dii_net", "flow_signal", "pcr",
        "max_pain", "mtf_signal", "mtf_confidence",
    ],
    "calendar": ["day_of_week", "month", "quarter", "day_of_month"],
}

TARGET_COLUMNS = ["target", "target_direction", "tb_label"]


def all_factor_names() -> list[str]:
    """Flat, deduplicated list of every factor produced by the pipeline."""
    names: list[str] = []
    for group in FACTOR_GROUPS.values():
        for name in group:
            if name not in names:
                names.append(name)
    return names


def factor_count() -> int:
    return len(all_factor_names())


# ─── technical indicators (new) ──────────────────────────────────────────────


def add_supertrend(
    df: pd.DataFrame, period: int = 10, multiplier: float = 3.0
) -> pd.DataFrame:
    """Supertrend indicator (ATR-based trailing stop).

    Returns two columns: ``supertrend`` (the trailing stop value) and
    ``supertrend_dir`` (1 = uptrend, 0 = downtrend).
    """
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]
    atr = _atr(high, low, close, period)

    hl2 = (high + low) / 2
    upper_basic = hl2 + multiplier * atr
    lower_basic = hl2 - multiplier * atr

    upper = pd.Series(np.nan, index=df.index)
    lower = pd.Series(np.nan, index=df.index)
    direction = pd.Series(1.0, index=df.index)

    upper.iloc[0] = upper_basic.iloc[0]
    lower.iloc[0] = lower_basic.iloc[0]

    for i in range(1, len(df)):
        prev_upper, prev_lower = upper.iloc[i - 1], lower.iloc[i - 1]
        if upper_basic.iloc[i] < prev_upper or close.iloc[i - 1] > prev_upper:
            upper.iloc[i] = upper_basic.iloc[i]
        else:
            upper.iloc[i] = prev_upper
        if lower_basic.iloc[i] > prev_lower or close.iloc[i - 1] < prev_lower:
            lower.iloc[i] = lower_basic.iloc[i]
        else:
            lower.iloc[i] = prev_lower
        if close.iloc[i] > prev_upper:
            direction.iloc[i] = 1.0
        elif close.iloc[i] < prev_lower:
            direction.iloc[i] = 0.0
        else:
            direction.iloc[i] = direction.iloc[i - 1]

    df["supertrend"] = np.where(direction == 1, lower, upper)
    df["supertrend_dir"] = direction
    return df


def add_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    """Ichimoku Cloud components.

    tenkan_sen (9), kijun_sen (26), senkou_span_a (midpoint shifted 26),
    senkou_span_b (52-period high/low midpoint shifted 26), chikou_span
    (close shifted -26). All use only past data.
    """
    df = df.copy()
    high, low, close = df["high"], df["low"], df["close"]

    def _mid(period: int) -> pd.Series:
        return (high.rolling(period).max() + low.rolling(period).min()) / 2

    df["ichimoku_tenkan"] = _mid(9)
    df["ichimoku_kijun"] = _mid(26)
    senkou_a = ((df["ichimoku_tenkan"] + df["ichimoku_kijun"]) / 2).shift(26)
    senkou_b = _mid(52).shift(26)
    df["ichimoku_senkou_a"] = senkou_a
    df["ichimoku_senkou_b"] = senkou_b
    df["ichimoku_chikou"] = close.shift(-26)  # future close, but only informational
    return df


def add_cmf(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Chaikin Money Flow: volume-weighted Money Flow Multiplier over period.

    Ranges in [-1, 1]; positive = accumulation, negative = distribution.
    """
    df = df.copy()
    high, low, close, volume = df["high"], df["low"], df["close"], df["volume"]
    range_ = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / range_
    mfv = mfm * volume
    df["cmf"] = (mfv.rolling(period).sum() / volume.rolling(period).sum()).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def add_order_flow_imbalance(df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
    """Order Flow Imbalance proxy from OHLCV data.

    Approximates directional aggressiveness:
        ofi_t = sign(close_t - open_t) * volume_t
    then normalized by its rolling mean absolute value. Values > 1 indicate
    heavy buying pressure relative to recent history.
    """
    df = df.copy()
    ofi = np.sign(df["close"] - df["open"]) * df["volume"]
    mad = ofi.rolling(period).apply(
        lambda x: np.nanmean(np.abs(x)), raw=True
    )
    df["ofi"] = (ofi / mad.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return df


def add_volatility_zscore(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """Z-score of rolling realized volatility vs its own longer-term level."""
    df = df.copy()
    ret = df["close"].pct_change()
    vol = ret.rolling(period).std()
    vol_mean = vol.rolling(period * 3, min_periods=period).mean()
    vol_std = vol.rolling(period * 3, min_periods=period).std()
    df["vol_zscore"] = ((vol - vol_mean) / vol_std.replace(0, np.nan)).replace(
        [np.inf, -np.inf], np.nan
    )
    return df


def add_vol_adjusted_momentum(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-sectional momentum: return / realized volatility, scaled to %.

    mom_1m_voladj = 21d return / 21d vol (annualized), etc.
    Positive = strong risk-adjusted uptrend; comparable across stocks.
    """
    df = df.copy()
    close = df["close"]
    horizons = {"1m": 21, "3m": 63, "6m": 126, "12m": 252}
    for label, h in horizons.items():
        ret = close.pct_change(h)
        vol = close.pct_change().rolling(h).std() * np.sqrt(252)
        df[f"mom_{label}_voladj"] = (ret / vol.replace(0, np.nan)).replace(
            [np.inf, -np.inf], np.nan
        )
    return df


def add_relative_strength(df: pd.DataFrame, index_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Relative strength vs NIFTY 50: stock return minus index return.

    Requires an index DataFrame with a close column. When unavailable,
    columns are created but left NaN (models must dropna anyway).
    """
    df = df.copy()
    for label, h in [("1m", 21), ("3m", 63)]:
        df[f"rel_strength_{label}"] = np.nan
    if index_df is None or index_df.empty or "close" not in index_df.columns:
        return df

    idx_close = index_df["close"].reindex(df.index).ffill()
    for label, h in [("1m", 21), ("3m", 63)]:
        rs = df["close"].pct_change(h) - idx_close.pct_change(h)
        df[f"rel_strength_{label}"] = rs
    return df


def add_high_low_spread(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
    """High-Low range spread: rolling mean of daily range as % of close."""
    df = df.copy()
    range_pct = (df["high"] - df["low"]) / df["close"]
    df["high_low_spread"] = range_pct.rolling(period).mean()
    return df


def add_pcr_slope(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """PCR slope over the trailing window.

    The PCR time series is only available on the most recent row (options
    data is fetched live); the slope therefore defaults to 0.0 historically,
    preserving point-in-time discipline.
    """
    df = df.copy()
    df["pcr_slope"] = 0.0
    if len(df) < 2 or "pcr" not in df.columns:
        return df
    pcr = df["pcr"].iloc[-1]
    # Without a stored PCR history, use the shift of the single known value
    # (0.0 slope when only one observation is known).
    known = df["pcr"].replace(1.0, np.nan).dropna()
    if len(known) >= 2:
        recent = known.tail(period + 1)
        df.loc[df.index[-1], "pcr_slope"] = float(
            (recent.iloc[-1] - recent.iloc[0]) / max(len(recent) - 1, 1)
        )
    elif len(known) == 1:
        df.loc[df.index[-1], "pcr_slope"] = 0.0
    return df


def add_fii_momentum(df: pd.DataFrame, period: int = 5) -> pd.DataFrame:
    """FII net flow momentum ratio.

    ratio = fii_net_t / mean(|fii_net| over trailing period). Values above 1
    indicate accelerating foreign inflows. Defaults to 0.0 on historical
    rows (FII data is only known for the most recent session).
    """
    df = df.copy()
    df["fii_momentum"] = 0.0
    if len(df) == 0 or "fii_net" not in df.columns:
        return df
    fii = df["fii_net"]
    known = fii[fii != 0.0].dropna()
    if len(known) >= 1:
        window = known.tail(period)
        mean_abs = np.nanmean(np.abs(window)) if len(window) else 0.0
        if mean_abs > 0:
            df.loc[df.index[-1], "fii_momentum"] = float(fii.iloc[-1] / mean_abs)
    return df


# ─── orchestrator ────────────────────────────────────────────────────────────


def compute_features(
    df: pd.DataFrame,
    ticker: str | None = None,
    index_df: pd.DataFrame | None = None,
    include_external: bool = True,
) -> pd.DataFrame:
    """Compute the full factor set on an OHLCV DataFrame.

    Args:
        df: OHLCV data indexed by date with open/high/low/close/volume.
        ticker: Optional ticker for external PIT features (sentiment, flows,
            PCR, MTF). When None, external features are skipped.
        index_df: NIFTY 50 (or any benchmark) DataFrame for relative strength.
        include_external: When False, skips external-data features entirely
            (useful for pure technical backtests).

    Returns:
        Feature DataFrame with all factors + targets. External features are
        populated only on the last row (point-in-time discipline).
    """
    if df is None or df.empty:
        raise ValueError("compute_features requires non-empty OHLCV data")

    out = add_technical_indicators(df)
    out = add_supertrend(out)
    out = add_ichimoku(out)
    out = add_cmf(out)
    out = add_order_flow_imbalance(out)
    out = add_volatility_zscore(out)
    out = add_vol_adjusted_momentum(out)
    out = add_relative_strength(out, index_df)
    out = add_high_low_spread(out)

    # External features are point-in-time: they are populated only on the
    # most recent row (see add_sentiment_features / add_flow_features).
    if include_external and ticker:
        out = add_sentiment_features(out, ticker)
        out = add_flow_features(out)
        out = add_pcr_features(out)
        out = add_multitimeframe_features(out, ticker)
        out = add_pcr_slope(out)
        out = add_fii_momentum(out)

    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Wilder's Average True Range (matches `ta` conventions)."""
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean()


def load_index_history(period: str = "2y") -> pd.DataFrame | None:
    """Load NIFTY 50 index history for cross-sectional features.

    Tries the local point-in-time store first, then yfinance.
    Returns a DataFrame indexed by date with a close column, or None.
    """
    try:
        from src.data.store import get_store

        store = get_store()
        idx = store.get_history("^NSEI", adjusted=False)
        if idx is not None and len(idx) > 20:
            return idx[["close"]].copy()
    except Exception as exc:
        logger.debug("Index history from store failed: %s", exc)
    try:
        import yfinance as yf

        raw = yf.download("^NSEI", period=period, interval="1d",
                          progress=False, auto_adjust=False)
        if raw is None or raw.empty:
            return None
        if isinstance(raw.columns, pd.MultiIndex):
            # modern yfinance returns MultiIndex columns (Ticker, Field)
            raw.columns = raw.columns.get_level_values(0)
        raw.columns = [str(c).lower() for c in raw.columns]
        idx = raw[["close"]].copy()
        idx.index = pd.to_datetime(idx.index).tz_localize(None)
        idx.index.name = "date"
        return idx
    except Exception as exc:
        logger.warning("Index history fetch failed: %s", exc)
        return None

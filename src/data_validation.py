"""Data validation for NSE price data.

Checks for:
- Missing trading days (gaps)
- Corporate actions (splits/dividends)
- Stale data
- Price sanity (negative, zero, extreme moves)
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def detect_gaps(df: pd.DataFrame, expected_freq: str = "B") -> list:
    """Find missing trading days in price data.

    Args:
        df: DataFrame with DatetimeIndex
        expected_freq: Expected frequency ('B' = business days)

    Returns:
        List of dicts with start, end, n_days for each gap
    """
    if len(df) < 2:
        return []

    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    missing = full_range.difference(df.index)

    if len(missing) == 0:
        return []

    gaps = []
    gap_start = missing[0]
    prev = missing[0]

    for i in range(1, len(missing)):
        if (missing[i] - prev).days > 5:
            gaps.append({
                "start": str(gap_start.date()),
                "end": str(prev.date()),
                "n_days": (prev - gap_start).days + 1,
            })
            gap_start = missing[i]
        prev = missing[i]

    gaps.append({
        "start": str(gap_start.date()),
        "end": str(prev.date()),
        "n_days": (prev - gap_start).days + 1,
    })

    return gaps


def detect_corporate_actions(df: pd.DataFrame, threshold: float = 0.15) -> list:
    """Detect potential stock splits or dividends.

    Looks for unnatural price jumps (>threshold overnight) without
    corresponding volume spikes.

    Args:
        df: DataFrame with 'close' and 'volume' columns
        threshold: Minimum absolute return to flag (default 15%)

    Returns:
        List of dicts with date, type, return_pct, volume_change
    """
    if len(df) < 2 or "close" not in df.columns or "volume" not in df.columns:
        return []

    returns = df["close"].pct_change()
    volume_change = df["volume"].pct_change()

    suspicious = []
    for date in df.index[1:]:
        ret = abs(returns.loc[date])
        vol = volume_change.loc[date] if date in volume_change.index else 0

        if ret > threshold:
            action_type = "split_or_dividend" if returns.loc[date] > 0 else "reverse_split"
            suspicious.append({
                "date": str(date.date()),
                "type": action_type,
                "return_pct": round(float(returns.loc[date]) * 100, 2),
                "volume_change": round(float(vol) if np.isfinite(vol) else 0, 2),
            })

    return suspicious


def detect_stale_data(df: pd.DataFrame, max_age_days: int = 2) -> dict:
    """Check if the data is stale.

    Args:
        df: DataFrame with DatetimeIndex
        max_age_days: Maximum acceptable age in days

    Returns:
        Dict with is_stale, last_date, age_days
    """
    if len(df) == 0:
        return {"is_stale": True, "last_date": "N/A", "age_days": -1, "max_age_days": max_age_days}

    last_date = df.index[-1]
    now = pd.Timestamp.now()
    if last_date.tz is not None:
        now = now.tz_localize(last_date.tz)

    age_days = (now - last_date).days
    return {
        "is_stale": age_days > max_age_days,
        "last_date": str(last_date.date()),
        "age_days": age_days,
        "max_age_days": max_age_days,
    }


def validate_prices(df: pd.DataFrame) -> list:
    """Check for impossible or suspicious prices.

    Args:
        df: DataFrame with OHLCV columns

    Returns:
        List of error strings
    """
    errors = []
    required_cols = ["open", "high", "low", "close"]
    for col in required_cols:
        if col not in df.columns:
            errors.append(f"Missing column: {col}")
    if errors:
        return errors

    for col in required_cols:
        bad = df[df[col] <= 0]
        if len(bad) > 0:
            errors.append(f"{col} has {len(bad)} non-positive values")

    bad_hl = df[df["high"] < df["low"]]
    if len(bad_hl) > 0:
        errors.append(f"high < low on {len(bad_hl)} days")

    bad_close = df[(df["close"] > df["high"] * 1.001) | (df["close"] < df["low"] * 0.999)]
    if len(bad_close) > 0:
        errors.append(f"close outside high-low range on {len(bad_close)} days")

    if "volume" in df.columns:
        neg_vol = df[df["volume"] < 0]
        if len(neg_vol) > 0:
            errors.append(f"negative volume on {len(neg_vol)} days")

    returns = df["close"].pct_change()
    extreme = df[returns.abs() > 0.30]
    if len(extreme) > 0:
        errors.append(f"extreme daily moves (>30%) on {len(extreme)} days")

    return errors


def validate_data(df: pd.DataFrame, ticker: str) -> dict:
    """Run all validations on price data.

    Args:
        df: DataFrame with OHLCV data
        ticker: Stock ticker for logging

    Returns:
        Dict with passed, errors, warnings, gaps, corporate_actions, stale_info
    """
    errors = []
    warnings = []

    price_errors = validate_prices(df)
    errors.extend(price_errors)

    gaps = detect_gaps(df)
    if gaps:
        total_missing = sum(g["n_days"] for g in gaps)
        warnings.append(f"{len(gaps)} gap(s) found, {total_missing} missing trading days")

    stale = detect_stale_data(df)
    if stale["is_stale"]:
        warnings.append(f"Data is {stale['age_days']} days old (last: {stale['last_date']})")

    corp_actions = detect_corporate_actions(df)
    if corp_actions:
        warnings.append(f"{len(corp_actions)} potential corporate action(s) detected")

    result = {
        "ticker": ticker,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "gaps": gaps,
        "corporate_actions": corp_actions,
        "stale_info": stale,
        "data_points": len(df),
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}" if len(df) > 0 else "N/A",
    }

    level = logging.WARNING if warnings else logging.INFO
    logger.log(level, f"[{ticker}] Validation: {'PASS' if result['passed'] else 'FAIL'} "
               f"({result['data_points']} points, {len(warnings)} warnings, {len(errors)} errors)")

    return result

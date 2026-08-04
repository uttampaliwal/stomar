import json
import tempfile
import yfinance as yf
import pandas as pd
from datetime import datetime
import os
import time
import logging
import threading

from src.data.free_data import fetch_free_historical_data
from src.data.resilience import retry_with_backoff, yf_breaker

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)

_fetch_cache: dict[str, tuple[float, pd.DataFrame]] = {}
_fetch_cache_lock = threading.Lock()
_FETCH_CACHE_TTL = 600  # 10 minutes in-memory cache

# P2.1/P2.4: records of automatic data repairs and per-ticker fetch health
_MONITORING_DIR = os.path.join(DATA_DIR, "monitoring")
_DATA_FIXES_PATH = os.path.join(_MONITORING_DIR, "data_fixes.json")
_FETCH_STATS_PATH = os.path.join(_MONITORING_DIR, "fetch_stats.json")
_MAX_FIXES_KEPT = 200


def _log_data_fix(ticker: str, kind: str, detail: str):
    """Append an automatic data repair to data/monitoring/data_fixes.json."""
    try:
        os.makedirs(_MONITORING_DIR, exist_ok=True)
        fixes = []
        if os.path.exists(_DATA_FIXES_PATH):
            with open(_DATA_FIXES_PATH) as f:
                fixes = json.load(f)
        fixes.append({
            "ticker": ticker,
            "kind": kind,
            "detail": detail,
            "timestamp": datetime.now().isoformat(),
        })
        fixes = fixes[-_MAX_FIXES_KEPT:]
        fd, tmp = tempfile.mkstemp(dir=_MONITORING_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(fixes, f, indent=2)
        os.replace(tmp, _DATA_FIXES_PATH)
        logger.info("Data fix recorded | ticker=%s kind=%s | %s", ticker, kind, detail)
    except Exception as e:
        logger.warning("Failed to record data fix: %s", e)


def _record_fetch(ticker: str, ok: bool, detail: str = ""):
    """Track per-ticker fetch success rate (P2.4)."""
    try:
        os.makedirs(_MONITORING_DIR, exist_ok=True)
        stats = {}
        if os.path.exists(_FETCH_STATS_PATH):
            with open(_FETCH_STATS_PATH) as f:
                stats = json.load(f)
        entry = stats.get(ticker, {"successes": 0, "failures": 0})
        if ok:
            entry["successes"] += 1
        else:
            entry["failures"] += 1
        entry["last_attempt"] = datetime.now().isoformat()
        entry["last_detail"] = detail[:200]
        entry["success_rate"] = entry["successes"] / (entry["successes"] + entry["failures"])
        stats[ticker] = entry
        fd, tmp = tempfile.mkstemp(dir=_MONITORING_DIR, suffix=".tmp")
        with os.fdopen(fd, "w") as f:
            json.dump(stats, f, indent=2)
        os.replace(tmp, _FETCH_STATS_PATH)
    except Exception as e:
        logger.warning("Failed to record fetch stats: %s", e)


def _cache_get(key: str) -> pd.DataFrame | None:
    with _fetch_cache_lock:
        if key in _fetch_cache:
            ts, df = _fetch_cache[key]
            if time.time() - ts < _FETCH_CACHE_TTL:
                return df
            del _fetch_cache[key]
    return None


def _cache_put(key: str, df: pd.DataFrame):
    with _fetch_cache_lock:
        _fetch_cache[key] = (time.time(), df)

NSE_STOCKS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "BAJFINANCE.NS", "LT.NS", "WIPRO.NS", "AXISBANK.NS", "TITAN.NS",
    "MARUTI.NS", "SUNPHARMA.NS", "ASIANPAINT.NS", "NTPC.NS", "ONGC.NS",
]

os.makedirs(DATA_DIR, exist_ok=True)

# Per-ticker locks: serialize fetch_stock_data so two concurrent requests
# for the same ticker cannot both miss the cache and download redundantly.
_ticker_fetch_locks: dict[str, threading.Lock] = {}
_ticker_fetch_locks_guard = threading.Lock()


def _get_ticker_lock(ticker: str) -> threading.Lock:
    with _ticker_fetch_locks_guard:
        lock = _ticker_fetch_locks.get(ticker)
        if lock is None:
            lock = _ticker_fetch_locks[ticker] = threading.Lock()
        return lock


def fetch_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    with _get_ticker_lock(ticker):
        return _fetch_stock_data_locked(ticker, period, interval, force_refresh)


def _fetch_stock_data_locked(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> pd.DataFrame:
    cache_key = f"{ticker}:{period}:{interval}"

    if not force_refresh:
        cached = _cache_get(cache_key)
        if cached is not None:
            return cached

    cache_path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")

    if not force_refresh and os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        last_date = df.index[-1]
        if last_date.tz is not None:
            now = pd.Timestamp.now(tz=last_date.tz)
        else:
            now = pd.Timestamp.now()
        # Refresh if data is stale (>2 days old) OR too short (< expected rows for period)
        expected_min_rows = {"1y": 200, "2y": 400, "3y": 600, "5y": 1000}
        min_rows = expected_min_rows.get(period, 200)
        if last_date >= now.normalize() - pd.Timedelta(days=2) and len(df) >= min_rows:
            _cache_put(cache_key, df)
            return df

    try:
        df = fetch_free_historical_data(ticker, period=period, interval=interval)
        if df is None or df.empty:
            raise ValueError("No data returned")
        df = df.set_index("date")
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        df = df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]
    except Exception:
        stock = yf.Ticker(ticker)
        df = yf_breaker.call(stock.history, period=period, interval=interval)
        if df.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"

    df.to_parquet(cache_path)
    _cache_put(cache_key, df)
    return df


@retry_with_backoff(max_retries=2, base_delay=1.0)
def get_live_price(ticker: str) -> float | None:
    """Fetch the latest live price for a ticker.

    Returns the price on success, None on failure. Callers must
    check for None before using the price to avoid zero-price trades.
    """
    stock = yf.Ticker(ticker)
    data = yf_breaker.call(stock.history, period="1d", interval="1m")
    if data.empty or "Close" not in data.columns:
        logger.warning("No live price data for %s", ticker)
        return None
    return float(data["Close"].iloc[-1])


def get_market_status() -> str:
    """Check NSE market status. Uses IST timezone to be server-agnostic."""
    try:
        from zoneinfo import ZoneInfo
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
    except Exception:
        # Fallback for environments without zoneinfo (add offset manually)
        from datetime import timezone, timedelta
        ist = timezone(timedelta(hours=5, minutes=30))
        now = datetime.now(ist)

    if now.weekday() >= 5:
        return "Closed (Weekend)"
    market_open = now.replace(hour=9, minute=15, second=0, microsecond=0)
    market_close = now.replace(hour=15, minute=30, second=0, microsecond=0)
    if market_open <= now <= market_close:
        return "Open"
    return "Closed"


def fetch_with_validation(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict:
    """Fetch stock data with validation.

    Returns:
        Dict with df, validation, ticker
    """
    from src.data.data_validation import validate_data

    df = fetch_stock_data(ticker, period=period, interval=interval, force_refresh=force_refresh)
    validation = validate_data(df, ticker)

    return {
        "df": df,
        "validation": validation,
        "ticker": ticker,
    }


def fetch_validated_stock_data(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    force_refresh: bool = False,
) -> dict:
    """Fetch + validate + auto-repair price data (P2.1/P2.2/P2.4).

    Pipeline:
      1. Fetch (cached), then validate for gaps / corporate actions.
      2. If gaps detected, re-fetch with an extended period to fill them.
      3. If gaps persist, forward-fill only non-price columns (volume,
         indicators) — never OHLC, which would fabricate prices.
      4. If suspicious overnight moves are found, force one adjusted
         re-fetch and log it (split/dividend handling).
      5. Record every repair in data_fixes.json and fetch health in
         fetch_stats.json.

    Returns:
        Dict with df, validation, fixes (list of logged repairs).
    """
    from src.data.data_validation import validate_data

    df = fetch_stock_data(ticker, period=period, interval=interval, force_refresh=force_refresh)
    if df is None or len(df) < 2:
        _record_fetch(ticker, ok=False, detail="no data returned")
        return {"df": df, "validation": {"passed": False, "errors": ["no data"]}, "fixes": []}

    validation = validate_data(df, ticker)
    fixes: list[dict] = []

    # ── P2.2: suspicious corporate-action moves → force adjusted re-fetch ──
    if validation.get("corporate_actions"):
        try:
            df2 = fetch_stock_data(ticker, period=period, interval=interval, force_refresh=True)
            validation2 = validate_data(df2, ticker)
            if not validation2.get("corporate_actions"):
                _log_data_fix(ticker, "corporate_action",
                              f"forced adjusted re-fetch; cleared {len(validation['corporate_actions'])} suspect moves")
                fixes.append("corporate_action")
                df, validation = df2, validation2
            elif len(df2) >= len(df):
                df = df2
                validation = validation2
        except Exception as e:
            logger.warning("Corporate-action re-fetch failed for %s: %s", ticker, e)

    # ── P2.1: gap filling via extended-period re-fetch ─────────────────────
    gaps = validation.get("gaps") or []
    if gaps:
        extended = {"1y": "2y", "2y": "5y", "3y": "5y", "5y": "5y"}.get(period, "5y")
        try:
            df_ext = fetch_stock_data(ticker, period=extended, interval=interval, force_refresh=True)
            if df_ext is not None and len(df_ext) >= len(df):
                df = df_ext
                validation = validate_data(df, ticker)
                gaps_after = validation.get("gaps") or []
                if len(gaps_after) < len(gaps):
                    _log_data_fix(ticker, "gap_fill",
                                  f"re-fetched with extended period {extended}: "
                                  f"gaps {sum(g['n_days'] for g in gaps)} -> {sum(g['n_days'] for g in gaps_after)}")
                    fixes.append("gap_fill")
        except Exception as e:
            logger.warning("Extended re-fetch failed for %s: %s", ticker, e)

    # ── P2.1: final fallback — forward-fill non-price columns only ─────────
    remaining = validation.get("gaps") or []
    if remaining and len(df) >= 2:
        non_price = [c for c in df.columns if c not in ("open", "high", "low", "close")]
        if non_price:
            filled = df[non_price].ffill()
            if filled.isna().sum().sum() == 0 or df[non_price].isna().any().any():
                df[non_price] = filled
                _log_data_fix(ticker, "gap_ffill",
                              f"forward-filled non-price columns {non_price} "
                              f"({sum(g['n_days'] for g in remaining)} missing days; OHLC untouched)")
                fixes.append("gap_ffill")
        # Holiday gaps (P2.3) are expected — clear them from the warning list
        real_gaps = [g for g in remaining
                     if g["start"] != g["end"] or not _is_holiday_window(g)]
        if not real_gaps:
            validation["gaps"] = []
            validation["warnings"] = [w for w in validation["warnings"] if "gap" not in w]

    _record_fetch(ticker, ok=True, detail=f"{len(df)} rows, {len(fixes)} fixes")
    return {"df": df, "validation": validation, "fixes": fixes}


def _is_holiday_window(gap: dict) -> bool:
    """True if every date in a 1-day gap is an NSE holiday (P2.3)."""
    from src.core.calendar import is_trading_day
    from datetime import datetime as _dt
    try:
        start = _dt.strptime(gap["start"], "%Y-%m-%d")
        end = _dt.strptime(gap["end"], "%Y-%m-%d")
        from datetime import timedelta
        current = start
        while current <= end:
            if is_trading_day(current):
                return False
            current += timedelta(days=1)
        return True
    except (ValueError, KeyError):
        return False

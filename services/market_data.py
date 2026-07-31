"""Free real-time market data engine for Indian equities (NSE/BSE).

Single entry point for candlestick data (1m, 5m, 15m, 1d), live quotes,
daily returns and on-the-fly technical indicators:

    RSI(14), MACD(12,26,9), ATR(14), Bollinger Bands(20, 2), Volume Z-Score.

Sources are tried in fallback order per interval:

    intraday (1m/5m/15m):  yfinance  →  local DuckDB store (live engine
                           minute bars, resampled)  →  parquet cache
    daily (1d):            yfinance  →  local DuckDB store  →  NSE archives

The yfinance source is rate-limited and protected by the shared circuit
breaker; every provider result is tagged with its source name so callers
can see which feed served the data.

Usage:
    from services.market_data import market_data_service as mds
    candles = mds.get_candles("RELIANCE.NS", interval="15m")
    quote   = mds.get_quote("RELIANCE.NS")
    rets    = mds.get_daily_returns("RELIANCE.NS", days=30)
"""

from __future__ import annotations

import logging
import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd

from src.data.resilience import RateLimiter, yf_breaker
from src.data.store import get_store
from src.data.universes import yf_ticker
from src.signals.multitimeframe import MTF_CACHE_DIR

logger = logging.getLogger(__name__)

SUPPORTED_INTERVALS = ("1m", "5m", "15m", "1d")

# yfinance caps each intraday interval at a fixed lookback window.
INTERVAL_PERIODS = {
    "1m": "5d",
    "5m": "60d",
    "15m": "60d",
    "1d": "2y",
}

# Store table name (or resample target) per interval.
STORE_INTERVALS = {
    "1m": "minute",
    "5m": "5min",
    "15m": "15min",
}

# Parquet cache timeframes produced by src.signals.multitimeframe.
PARQUET_TIMEFRAMES = {"15m": "15m", "1d": "daily"}

_CACHE_TTL = 30.0          # in-memory candle cache TTL (seconds)
_MAX_LIMIT = 20000
_DEFAULT_LIMIT = 500

_YF_LIMITER = RateLimiter(rate=5.0, burst=8, name="market-data-yf")

_OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class CandleProvider(ABC):
    """A free source of OHLCV candles for a set of intervals."""

    name: str = "base"
    supports: set[str] = set()

    @abstractmethod
    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        """Return tz-naive OHLCV bars indexed by timestamp. Raises on failure."""

    def is_available(self) -> bool:
        return True


# ─── providers ──────────────────────────────────────────────────────────────


class YahooCandleProvider(CandleProvider):
    """Primary source: Yahoo Finance chart history (covers all intervals)."""

    name = "yahoo"
    supports = set(SUPPORTED_INTERVALS)

    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        import yfinance as yf

        ticker = yf_ticker(symbol)
        period = INTERVAL_PERIODS.get(interval, "60d")
        with _YF_LIMITER.acquire(ticker):
            df = yf_breaker.call(
                lambda: yf.Ticker(ticker).history(
                    period=period, interval=interval, auto_adjust=False
                )
            )
        if df is None or df.empty:
            raise ValueError(f"yfinance returned no {interval} bars for {ticker}")
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        return df[[c for c in _OHLCV_COLS if c in df.columns]]


class StoreCandleProvider(CandleProvider):
    """Fallback: point-in-time DuckDB store (live-engine minute bars / daily)."""

    name = "store"
    supports = set(STORE_INTERVALS) | {"1d"}

    def __init__(self, store=None):
        self.store = store or get_store()

    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        if interval == "1d":
            df = self.store.get_history(symbol, interval="daily")
        else:
            df = self.store.get_history(symbol, interval="minute")
            if df.empty:
                raise ValueError(f"No minute bars in store for {symbol}")
            if "ts" in df.columns:
                df = df.set_index("ts").sort_index()
            df = _resample(df, STORE_INTERVALS[interval])
        if df.empty:
            raise ValueError(f"No {interval} bars in store for {symbol}")
        return df[[c for c in _OHLCV_COLS if c in df.columns]]


class ParquetCandleProvider(CandleProvider):
    """Last resort: parquet caches written by the MTF pipeline."""

    name = "parquet"
    supports = set(PARQUET_TIMEFRAMES)

    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        tf = PARQUET_TIMEFRAMES[interval]
        path = os.path.join(MTF_CACHE_DIR, f"mtf_{symbol.replace('.', '_')}_{tf}.parquet")
        if not os.path.exists(path):
            raise ValueError(f"No parquet cache at {path}")
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index).tz_localize(None) if df.index.tz is not None else pd.to_datetime(df.index)
        return df[[c for c in _OHLCV_COLS if c in df.columns]]


class NseArchiveCandleProvider(CandleProvider):
    """Daily fallback: NSE public bhavcopy archives (best-effort, single month)."""

    name = "nse"
    supports = {"1d"}

    def fetch(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        if interval != "1d":
            raise ValueError("NSE archives only provide daily bars")
        from src.data.data_sources import NSEArchiveSource

        df = NSEArchiveSource().fetch(symbol, period="1mo", interval="1d")
        df.index = pd.to_datetime(df.index)
        return df[[c for c in _OHLCV_COLS if c in df.columns]]


# ─── provider chains ────────────────────────────────────────────────────────


def _chain_for(interval: str, providers: list[CandleProvider]) -> list[CandleProvider]:
    """Return the providers that support *interval*, in configured order."""
    return [p for p in providers if interval in p.supports]


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample minute bars (index = ts) up to 5m/15m."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    out = df.resample(rule).agg(agg).dropna(subset=["open", "close"])
    return out


# ─── technical indicators ───────────────────────────────────────────────────


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI(14), MACD(12,26,9), ATR(14), Bollinger Bands(20,2) and
    Volume Z-Score(20) columns to an OHLCV frame. Uses the `ta` library
    (project dependency) with pure-pandas volume z-score."""
    import ta

    out = df.copy()
    close, high, low, volume = out["close"], out["high"], out["low"], out["volume"]

    out["rsi"] = ta.momentum.rsi(close, window=14)
    out["macd"] = ta.trend.macd(close, window_slow=26, window_fast=12)
    out["macd_signal"] = ta.trend.macd_signal(
        close, window_slow=26, window_fast=12, window_sign=9
    )
    out["macd_hist"] = out["macd"] - out["macd_signal"]
    out["atr"] = ta.volatility.average_true_range(high, low, close, window=14)
    out["bb_middle"] = ta.volatility.bollinger_mavg(close, window=20)
    out["bb_upper"] = ta.volatility.bollinger_hband(close, window=20, window_dev=2)
    out["bb_lower"] = ta.volatility.bollinger_lband(close, window=20, window_dev=2)
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / out["bb_middle"]

    vol_mean = volume.rolling(20).mean()
    vol_std = volume.rolling(20).std(ddof=0)
    out["volume_z"] = (volume - vol_mean) / vol_std.replace(0, pd.NA)
    return out


_INDICATOR_COLS = [
    "rsi", "macd", "macd_signal", "macd_hist", "atr",
    "bb_middle", "bb_upper", "bb_lower", "bb_width", "volume_z",
]


# ─── payload helpers ────────────────────────────────────────────────────────


def candles_to_payload(
    symbol: str,
    interval: str,
    df: pd.DataFrame,
    source: str,
    indicators: bool = True,
    limit: int = _DEFAULT_LIMIT,
) -> dict:
    """Serialize a candle frame into the chart-rendering payload."""
    df = df.dropna(subset=["close"]).tail(limit)
    cols = list(_OHLCV_COLS)
    if indicators:
        cols = cols + _INDICATOR_COLS
    bars = []
    for idx, row in df.iterrows():
        bar: dict = {"timestamp": str(idx)}
        for col in cols:
            val = row.get(col)
            if val is None or pd.isna(val):
                continue
            if col == "volume":
                bar[col] = int(val)
            else:
                bar[col] = round(float(val), 6)
        bars.append(bar)
    return {
        "symbol": symbol,
        "interval": interval,
        "source": source,
        "rows": len(bars),
        "indicators": indicators,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "bars": bars,
    }


# ─── service ────────────────────────────────────────────────────────────────


class MarketDataService:
    """Orchestrates candle providers with fallback, caching and quote access."""

    def __init__(
        self,
        providers: list[CandleProvider] | None = None,
        cache_ttl: float = _CACHE_TTL,
    ):
        self.providers = providers or self._default_providers()
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, pd.DataFrame, str]] = {}
        self._lock = threading.Lock()
        self._health: dict[str, bool] = {}

    @staticmethod
    def _default_providers() -> list[CandleProvider]:
        return [
            YahooCandleProvider(),
            StoreCandleProvider(),
            NseArchiveCandleProvider(),
            ParquetCandleProvider(),
        ]

    # ── candles with fallback ───────────────────────────────────────────

    def get_candles(
        self,
        symbol: str,
        interval: str = "15m",
        limit: int = _DEFAULT_LIMIT,
        indicators: bool = True,
        force: bool = False,
    ) -> dict:
        """Fetch candles for *interval*, trying providers in fallback order.

        Returns the full chart payload (bars + indicator columns). Raises
        RuntimeError when every source fails.
        """
        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(
                f"interval must be one of {SUPPORTED_INTERVALS}, got {interval!r}"
            )
        limit = max(1, min(limit, _MAX_LIMIT))

        df, source = self._fetch(symbol, interval, force=force)

        # stitch indicators only over the *full* fetched frame so warm-up
        # periods are correct, then trim to the requested limit
        if indicators:
            df = compute_indicators(df)
        return candles_to_payload(symbol, interval, df, source, indicators=indicators, limit=limit)

    def _fetch(self, symbol: str, interval: str, force: bool = False) -> tuple[pd.DataFrame, str]:
        cache_key = f"{symbol}:{interval}"
        if not force:
            with self._lock:
                cached = self._cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < self.cache_ttl:
                return cached[1], cached[2]

        errors: list[str] = []
        for provider in _chain_for(interval, self.providers):
            if not provider.is_available():
                continue
            try:
                df = provider.fetch(symbol, interval, limit=_DEFAULT_LIMIT)
                if df is None or df.empty:
                    raise ValueError("provider returned no rows")
                df = df.sort_index()
                df = df[~df.index.duplicated(keep="last")]
                with self._lock:
                    self._cache[cache_key] = (time.monotonic(), df, provider.name)
                self._health[provider.name] = True
                return df, provider.name
            except Exception as exc:
                errors.append(f"{provider.name}: {exc}")
                self._health[provider.name] = False
                logger.debug("candle fallback for %s %s (%s)", symbol, interval, errors[-1])
                continue

        raise RuntimeError(f"All candle sources failed for {symbol} ({interval}): {'; '.join(errors)}")

    # ── quote / returns / health ─────────────────────────────────────────

    def get_quote(self, symbol: str, force: bool = False) -> dict:
        """Current price via the live engine's NSE → Yahoo → Google → Kite chain."""
        from src.data.live import live_engine

        quote = live_engine.get_quote(symbol, force=force)
        return quote.to_dict()

    def get_daily_returns(self, symbol: str, days: int = 30) -> dict:
        """Daily close-to-close returns over the last *days* sessions."""
        days = max(2, min(days, _MAX_LIMIT))
        payload = self.get_candles(symbol, interval="1d", limit=days, indicators=False)
        bars = payload["bars"]
        series = []
        for prev, cur in zip(bars, bars[1:]):
            if not prev.get("close") or not cur.get("close"):
                continue
            series.append({
                "timestamp": cur["timestamp"],
                "close": cur["close"],
                "return_pct": round((cur["close"] / prev["close"] - 1.0) * 100.0, 4),
            })
        return {
            "symbol": symbol,
            "source": payload["source"],
            "days": len(series),
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "returns": series,
        }

    def sources_health(self) -> dict:
        """Per-source availability across the fallback chain."""
        with self._lock:
            health = dict(self._health)
            cached = {k: round(self.cache_ttl - (time.monotonic() - v[0]), 1)
                      for k, v in self._cache.items() if time.monotonic() - v[0] < self.cache_ttl}
        return {
            "providers": [
                {"name": p.name, "intervals": sorted(p.supports),
                 "last_status": health.get(p.name)}
                for p in self.providers
            ],
            "cache_entries": len(cached),
            "cache": cached,
        }


_market_data_service: MarketDataService | None = None
_service_lock = threading.Lock()


def get_market_data_service() -> MarketDataService:
    """Process-wide market data service singleton."""
    global _market_data_service
    with _service_lock:
        if _market_data_service is None:
            _market_data_service = MarketDataService()
        return _market_data_service


market_data_service = get_market_data_service()

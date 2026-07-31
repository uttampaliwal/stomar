"""Multi-threaded historical OHLCV ingestion for stock universes.

Sources (in fallback order):
    1. yfinance — primary (split/bonus/dividend aware via auto_adjust=False)
    2. NSE public archives — best-effort fallback
    3. nsepython equity history — best-effort fallback

Fetches are rate-limited per source, retried with exponential backoff, and
executed in parallel via a ThreadPoolExecutor. Results are written to the
point-in-time DuckDB store (raw bars) and mirrored to parquet cache files.

Usage:
    from src.data.historical import backfill_universe
    result = backfill_universe("nifty_50", period="2y", workers=6)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Iterable

import pandas as pd

from src.core.constants import DATA_DIR
from src.data.data_sources import NSEArchiveSource
from src.data.resilience import RateLimiter
from src.data.store import MarketDataStore, get_store
from src.data.universes import get_universe, yf_ticker

logger = logging.getLogger(__name__)

_YF_LIMITER = RateLimiter(rate=5.0, burst=8, name="yfinance")
_NSE_LIMITER = RateLimiter(rate=2.0, burst=4, name="nse")


def _fetch_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    with _YF_LIMITER.acquire(ticker):
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval, auto_adjust=False)
        if df.empty:
            raise ValueError(f"yfinance returned no data for {ticker}")
        df = df.rename(columns={c: c.lower() for c in df.columns})
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df.index.name = "date"
        return df[[c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]]


def _fetch_nse_archive(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if interval != "1d":
        raise ValueError("NSE archive only supports daily data")
    return NSEArchiveSource().fetch(ticker, period, interval)


def _fetch_nsepython(ticker: str, period: str, interval: str) -> pd.DataFrame:
    if interval != "1d":
        raise ValueError("nsepython equity_history only supports daily data")
    try:
        import nsepython  # type: ignore
    except ImportError as exc:
        raise ImportError("nsepython not installed") from exc
    symbol = ticker.upper().removesuffix(".NS")
    with _NSE_LIMITER.acquire(symbol):
        try:
            df = nsepython.equity_history(symbol, period=period, interval=interval)
        except TypeError:
            df = nsepython.equity_history(symbol)
    if df is None or df.empty:
        raise ValueError(f"nsepython returned no data for {ticker}")
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def fetch_ticker_with_fallback(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    sources: Iterable[Callable[[str, str, str], pd.DataFrame]] | None = None,
) -> dict:
    """Fetch one ticker trying each source in order.

    Returns:
        Dict with df, source, ticker, rows, error (error set when all failed).
    """
    if sources is None:
        sources = [_fetch_yfinance, _fetch_nse_archive, _fetch_nsepython]
    sources = list(sources)

    last_error: Exception | None = None
    for fetch_fn in sources:
        try:
            df = fetch_fn(ticker, period, interval)
            if df is None or df.empty:
                raise ValueError(f"source {fetch_fn.__name__} returned no rows")
            df = df.sort_index()
            df = df[~df.index.duplicated(keep="last")]
            return {
                "df": df, "source": fetch_fn.__name__, "ticker": ticker,
                "rows": len(df), "error": None,
            }
        except Exception as exc:
            last_error = exc
            logger.warning("%s failed for %s: %s", fetch_fn.__name__, ticker, exc)
    return {
        "df": None, "source": None, "ticker": ticker,
        "rows": 0, "error": str(last_error or "unknown error"),
    }


def fetch_universe(
    universe: str | list[str],
    period: str = "2y",
    interval: str = "1d",
    workers: int = 5,
    as_yf: bool = True,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, dict]:
    """Fetch history for a whole universe in parallel.

    Args:
        universe: Universe name ("nifty_50", ...) or a list of symbols.
        period: yfinance period string ("2y", "5y", "max").
        interval: bar interval ("1d", "1wk", "1mo", "1h", "15m", ...).
        workers: ThreadPoolExecutor size.
        as_yf: When universe is a name, request .NS tickers.
        progress: Optional callback(result) after each ticker completes.

    Returns:
        Mapping ticker -> result dict (df, source, rows, error).
    """
    if isinstance(universe, str):
        tickers = get_universe(universe, as_yf=as_yf)
    else:
        tickers = [yf_ticker(t) for t in universe] if as_yf else list(universe)

    results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hist-fetch") as pool:
        futures = {
            pool.submit(fetch_ticker_with_fallback, t, period, interval): t
            for t in tickers
        }
        for future in as_completed(futures):
            ticker = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"df": None, "source": None, "ticker": ticker, "rows": 0, "error": str(exc)}
            results[ticker] = result
            if progress:
                progress(result)
    return results


def backfill_universe(
    universe: str | list[str],
    period: str = "2y",
    interval: str = "1d",
    workers: int = 5,
    store: MarketDataStore | None = None,
    save_parquet: bool = True,
) -> dict:
    """Fetch a universe and persist raw bars into the point-in-time store.

    Also mirrors each symbol to ``data/<SYMBOL>.parquet`` (existing project
    convention) when save_parquet is True.

    Returns:
        Dict with summary: total, ok, failed, failures, upserted.
    """
    store = store or get_store()
    results = fetch_universe(universe, period, interval, workers=workers)

    ok = 0
    failed = 0
    failures: dict[str, str] = {}
    upserted = 0
    for ticker, result in results.items():
        if result["df"] is None or result["error"]:
            failed += 1
            failures[ticker] = result.get("error", "unknown")
            continue
        df = result["df"]
        try:
            rows = store.upsert_daily(ticker, df, source=result["source"])
            upserted += rows
            if save_parquet:
                path = os.path.join(DATA_DIR, f"{ticker.replace('.', '_')}.parquet")
                df.to_parquet(path)
            ok += 1
        except Exception as exc:
            failed += 1
            failures[ticker] = str(exc)

    summary = {
        "total": len(results),
        "ok": ok,
        "failed": failed,
        "failures": failures,
        "rows_upserted": upserted,
    }
    logger.info("Backfill %s: %s", universe, summary)
    return summary

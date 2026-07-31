"""Point-in-time market data store backed by DuckDB.

Stores RAW (unadjusted) OHLCV bars, corporate actions (splits, bonuses,
dividends), options data and quote snapshots. Adjustments are computed at
query time with an optional ``as_of`` date so that only corporate actions
known on that date are applied — eliminating look-ahead bias.

Usage:
    from src.data.store import get_store
    store = get_store()
    store.upsert_daily("RELIANCE.NS", df)
    hist = store.get_history("RELIANCE.NS", start="2024-01-01", as_of="2024-12-31")
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from contextlib import contextmanager
from datetime import date, datetime
from typing import Iterable, Iterator, Sequence

import duckdb
import pandas as pd

from src.core.constants import DATA_DIR

logger = logging.getLogger(__name__)

MARKET_DATA_DB = os.path.join(DATA_DIR, "market_data.db")

_OHLCV_COLUMNS = ["symbol", "date", "open", "high", "low", "close", "volume", "source"]
_MINUTE_COLUMNS = ["symbol", "ts", "open", "high", "low", "close", "volume", "source"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv_daily (
    symbol     VARCHAR NOT NULL,
    date       DATE    NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     BIGINT,
    source     VARCHAR,
    fetched_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, date)
);

CREATE TABLE IF NOT EXISTS minute_bars (
    symbol     VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    open       DOUBLE,
    high       DOUBLE,
    low        DOUBLE,
    close      DOUBLE,
    volume     BIGINT,
    source     VARCHAR,
    fetched_at TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS corporate_actions (
    symbol             VARCHAR NOT NULL,
    ex_date            DATE    NOT NULL,
    action_type        VARCHAR NOT NULL,
    ratio              DOUBLE,
    dividend_per_share DOUBLE,
    face_value         DOUBLE,
    source             VARCHAR,
    fetched_at         TIMESTAMP DEFAULT current_timestamp,
    PRIMARY KEY (symbol, ex_date, action_type)
);

CREATE TABLE IF NOT EXISTS options_chain (
    symbol      VARCHAR NOT NULL,
    expiry      DATE,
    strike      DOUBLE,
    option_type VARCHAR,
    open_interest BIGINT,
    volume      BIGINT,
    last_price  DOUBLE,
    snapshot_ts TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, expiry, strike, option_type, snapshot_ts)
);

CREATE TABLE IF NOT EXISTS quote_snapshots (
    symbol     VARCHAR NOT NULL,
    ts         TIMESTAMP NOT NULL,
    price      DOUBLE,
    change_pct DOUBLE,
    volume     BIGINT,
    bid        DOUBLE,
    ask        DOUBLE,
    source     VARCHAR,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS depth_snapshots (
    symbol      VARCHAR NOT NULL,
    ts          TIMESTAMP NOT NULL,
    depth_json  VARCHAR,
    source      VARCHAR,
    PRIMARY KEY (symbol, ts)
);

CREATE TABLE IF NOT EXISTS universe_snapshot (
    universe VARCHAR NOT NULL,
    symbol   VARCHAR NOT NULL,
    added    DATE DEFAULT current_date,
    PRIMARY KEY (universe, symbol)
);
"""

# SPLIT/BONUS ratio semantics: shares after per share held before.
#   5:1 split  -> ratio = 5.0
#   1:2 bonus  -> ratio = 1.5
# Price bars before ex_date are multiplied by ratio; volumes multiplied by ratio.


class MarketDataStore:
    """DuckDB-backed point-in-time market data store."""

    def __init__(self, db_path: str = MARKET_DATA_DB):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._conn = self._connect_with_retry()
        self._conn.execute(_SCHEMA)
        self._lock = threading.RLock()
        logger.info("Market data store initialised at %s", db_path)

    def _connect_with_retry(self, attempts: int = 5, base_delay: float = 0.5) -> duckdb.DuckDBPyConnection:
        """Open the DuckDB file, retrying on lock conflicts.

        DuckDB allows a single writer process; if another process (e.g. a
        running pipeline) holds the file lock, wait briefly rather than
        failing immediately.
        """
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                return duckdb.connect(self.db_path)
            except Exception as exc:
                last_exc = exc
                if "lock" not in str(exc).lower():
                    raise
                logger.warning(
                    "Market data DB lock held by another process (attempt %d/%d): %s",
                    attempt, attempts, exc,
                )
                time.sleep(base_delay * attempt)
        raise RuntimeError(
            f"Cannot open market data store {self.db_path}: another process "
            f"holds the DuckDB file lock. Details: {last_exc}"
        )

    # ── lifecycle ────────────────────────────────────────────────────────

    def close(self):
        with self._lock:
            try:
                self._conn.close()
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @contextmanager
    def _tx(self) -> Iterator[duckdb.DuckDBPyConnection]:
        with self._lock:
            yield self._conn

    def query(self, sql: str, params: Sequence | None = None) -> pd.DataFrame:
        """Run a read-only query and return a DataFrame."""
        with self._tx() as conn:
            return conn.execute(sql, params or []).df()

    # ── daily bars ───────────────────────────────────────────────────────

    def upsert_daily(self, symbol: str, df: pd.DataFrame, source: str = "yfinance") -> int:
        """Insert or update raw daily OHLCV bars (dedup on symbol+date).

        Args:
            symbol: Ticker, e.g. "RELIANCE.NS".
            df: DataFrame with date index and open/high/low/close/volume.

        Returns:
            Number of rows upserted.
        """
        rows = _df_to_rows(df, symbol, source, daily=True)
        if not rows:
            return 0
        with self._tx() as conn:
            conn.executemany(
                f"""
                INSERT INTO ohlcv_daily ({", ".join(_OHLCV_COLUMNS)})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, date) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, source = EXCLUDED.source,
                    fetched_at = now()
                """,
                rows,
            )
        return len(rows)

    def upsert_minute_bars(self, symbol: str, df: pd.DataFrame, source: str = "nse") -> int:
        rows = _df_to_rows(df, symbol, source, daily=False)
        if not rows:
            return 0
        with self._tx() as conn:
            conn.executemany(
                f"""
                INSERT INTO minute_bars ({", ".join(_MINUTE_COLUMNS)})
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, ts) DO UPDATE SET
                    open = EXCLUDED.open, high = EXCLUDED.high,
                    low = EXCLUDED.low, close = EXCLUDED.close,
                    volume = EXCLUDED.volume, source = EXCLUDED.source,
                    fetched_at = now()
                """,
                rows,
            )
        return len(rows)

    def latest_bar(self, symbol: str) -> pd.DataFrame | None:
        df = self.query(
            "SELECT * FROM ohlcv_daily WHERE symbol = ? ORDER BY date DESC LIMIT 1",
            [symbol],
        )
        return df if len(df) else None

    # ── corporate actions ────────────────────────────────────────────────

    def record_corporate_action(
        self,
        symbol: str,
        ex_date: date | str,
        action_type: str,
        ratio: float | None = None,
        dividend_per_share: float | None = None,
        face_value: float | None = None,
        source: str = "manual",
    ):
        action_type = action_type.lower().strip()
        if action_type not in ("split", "bonus", "dividend"):
            raise ValueError(f"action_type must be split|bonus|dividend, got {action_type!r}")
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO corporate_actions
                    (symbol, ex_date, action_type, ratio, dividend_per_share, face_value, source)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, ex_date, action_type) DO UPDATE SET
                    ratio = COALESCE(EXCLUDED.ratio, corporate_actions.ratio),
                    dividend_per_share = COALESCE(EXCLUDED.dividend_per_share, corporate_actions.dividend_per_share),
                    face_value = COALESCE(EXCLUDED.face_value, corporate_actions.face_value),
                    source = EXCLUDED.source,
                    fetched_at = now()
                """,
                [symbol, str(ex_date), action_type, ratio, dividend_per_share, face_value, source],
            )

    def get_corporate_actions(self, symbol: str, as_of: date | str | None = None) -> pd.DataFrame:
        sql = "SELECT * FROM corporate_actions WHERE symbol = ?"
        params: list = [symbol]
        if as_of is not None:
            sql += " AND ex_date <= ?"
            params.append(str(as_of))
        return self.query(sql + " ORDER BY ex_date", params)

    def sync_corporate_actions_from_yfinance(self, symbol: str, source: str = "yfinance") -> int:
        """Import split/bonus and dividend history from Yahoo Finance.

        Yahoo records bonuses and splits as 'X:Y' split-style events; both are
        stored with their multiplier ratio. Dividends are stored per share.

        Returns:
            Number of actions upserted (0 if the symbol is unavailable).
        """
        import yfinance as yf

        stock = yf.Ticker(symbol)
        count = 0

        splits = stock.splits
        if splits is not None and len(splits):
            for ex_date, raw in splits.items():
                ex_date = pd.Timestamp(ex_date).date()
                ratio = _parse_split_ratio(raw)
                if ratio:
                    self.record_corporate_action(
                        symbol, ex_date, "split", ratio=ratio, source=source
                    )
                    count += 1

        dividends = stock.dividends
        if dividends is not None and len(dividends):
            for ex_date, amount in dividends.items():
                amount = float(amount)
                if amount > 0:
                    ex_date = pd.Timestamp(ex_date).date()
                    self.record_corporate_action(
                        symbol, ex_date, "dividend", dividend_per_share=amount, source=source
                    )
                    count += 1
        return count

    def sync_corporate_actions_from_nse(self, symbol: str) -> int:
        """Best-effort import of corporate actions from NSE's public API."""
        nse_symbol = symbol.upper().removesuffix(".NS")
        try:
            import requests

            resp = requests.get(
                "https://www.nseindia.com/api/corporates-corporateActions",
                params={"index": "equities"},
                headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"},
                timeout=20,
            )
            resp.raise_for_status()
            count = 0
            for row in resp.json():
                if row.get("symbol") != nse_symbol:
                    continue
                purpose = str(row.get("purpose", "")).lower()
                action_type = "dividend" if "dividend" in purpose else None
                if action_type is None and ("bonus" in purpose or "split" in purpose):
                    action_type = "bonus" if "bonus" in purpose else "split"
                if action_type is None:
                    continue
                try:
                    ex_date = pd.Timestamp(row["exdate"]).date()
                except (KeyError, ValueError):
                    continue
                self.record_corporate_action(
                    symbol, ex_date, action_type,
                    dividend_per_share=float(row.get("divPerShare") or 0) or None,
                    source="nse",
                )
                count += 1
            return count
        except Exception as exc:
            logger.warning("NSE corporate actions sync failed for %s: %s", symbol, exc)
            return 0

    # ── point-in-time history ────────────────────────────────────────────

    def get_history(
        self,
        symbol: str,
        start: str | date | None = None,
        end: str | date | None = None,
        as_of: str | date | None = None,
        adjusted: bool = True,
        interval: str = "daily",
    ) -> pd.DataFrame:
        """Fetch OHLCV with point-in-time adjustments.

        Only bars with ``date <= as_of`` and corporate actions with
        ``ex_date <= as_of`` are used — later events are invisible to the
        query, so no look-ahead bias is possible.

        Args:
            symbol: Ticker (e.g. "RELIANCE.NS").
            start/end: Date range (inclusive). Optional.
            as_of: Point-in-time cutoff (default: today).
            adjusted: Apply backward split/bonus/dividend adjustment.
            interval: "daily" or "minute".

        Returns:
            DataFrame indexed by date (daily) or ts (minute) with columns
            open/high/low/close/volume and (when adjusted) adjusted_* columns.
        """
        table = "ohlcv_daily" if interval == "daily" else "minute_bars"
        time_col = "date" if interval == "daily" else "ts"
        if interval not in ("daily", "minute"):
            raise ValueError(f"interval must be 'daily' or 'minute', got {interval!r}")

        where = "symbol = ?"
        params: list = [symbol]
        if start is not None:
            where += f" AND {time_col} >= ?"
            params.append(str(start))
        if end is not None:
            where += f" AND {time_col} <= ?"
            params.append(str(end))
        if as_of is not None:
            where += f" AND {time_col} <= ?"
            params.append(str(as_of))

        df = self.query(
            f"SELECT * FROM {table} WHERE {where} ORDER BY {time_col}", params
        )
        if df.empty:
            return df
        df = df.set_index(time_col).sort_index()

        if adjusted:
            events = self.get_corporate_actions(symbol, as_of=as_of)
            adjusted_df = adjust_ohlcv(df, events)
            for col in ["open", "high", "low", "close", "volume"]:
                if col in adjusted_df.columns:
                    df[f"adjusted_{col}"] = adjusted_df[col]
        return df

    def export_parquet(self, symbol: str, path: str, as_of: str | date | None = None):
        df = self.get_history(symbol, as_of=as_of)
        df.to_parquet(path)

    # ── live snapshots ───────────────────────────────────────────────────

    def record_quote(self, symbol: str, price: float, change_pct: float = 0.0,
                     volume: int = 0, bid: float | None = None, ask: float | None = None,
                     source: str = "nse", ts: datetime | None = None):
        ts = ts or datetime.now()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO quote_snapshots (symbol, ts, price, change_pct, volume, bid, ask, source)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [symbol, ts.isoformat(), float(price), float(change_pct),
                 int(volume or 0), bid, ask, source],
            )

    def record_depth(self, symbol: str, depth_json: dict, source: str = "nse",
                     ts: datetime | None = None):
        ts = ts or datetime.now()
        with self._tx() as conn:
            conn.execute(
                "INSERT INTO depth_snapshots (symbol, ts, depth_json, source) VALUES (?, ?, ?, ?)",
                [symbol, ts.isoformat(), json.dumps(depth_json), source],
            )

    def record_options_chain(
        self,
        symbol: str,
        chain_df: pd.DataFrame,
        snapshot_ts: datetime | None = None,
    ):
        """Persist an options chain snapshot.

        Args:
            symbol: Underlying, e.g. "NIFTY".
            chain_df: DataFrame with columns expiry, strike, option_type,
                      open_interest, volume, last_price.
        """
        if chain_df is None or chain_df.empty:
            return 0
        snapshot_ts = snapshot_ts or datetime.now()
        rows = []
        for _, row in chain_df.iterrows():
            rows.append((
                symbol,
                str(pd.Timestamp(row["expiry"]).date()) if pd.notna(row.get("expiry")) else None,
                float(row["strike"]),
                str(row["option_type"]).upper(),
                int(row.get("open_interest", 0) or 0),
                int(row.get("volume", 0) or 0),
                float(row["last_price"]) if pd.notna(row.get("last_price")) else None,
                snapshot_ts.isoformat(),
            ))
        with self._tx() as conn:
            conn.executemany(
                """
                INSERT INTO options_chain
                    (symbol, expiry, strike, option_type, open_interest, volume, last_price, snapshot_ts)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT (symbol, expiry, strike, option_type, snapshot_ts) DO UPDATE SET
                    open_interest = EXCLUDED.open_interest, volume = EXCLUDED.volume,
                    last_price = EXCLUDED.last_price
                """,
                rows,
            )
        return len(rows)

    # ── universe bookkeeping ─────────────────────────────────────────────

    def upsert_universe(self, name: str, symbols: Iterable[str]):
        with self._tx() as conn:
            conn.execute("DELETE FROM universe_snapshot WHERE universe = ?", [name])
            conn.executemany(
                "INSERT INTO universe_snapshot (universe, symbol) VALUES (?, ?)",
                [(name, s) for s in symbols],
            )

    def get_universe(self, name: str) -> list[str]:
        df = self.query(
            "SELECT symbol FROM universe_snapshot WHERE universe = ? ORDER BY symbol", [name]
        )
        return df["symbol"].tolist() if len(df) else []

    def available_symbols(self) -> list[str]:
        df = self.query(
            "SELECT DISTINCT symbol FROM ohlcv_daily ORDER BY symbol"
        )
        return df["symbol"].tolist() if len(df) else []

    def stats(self) -> dict:
        out = {}
        for table in ["ohlcv_daily", "minute_bars", "corporate_actions",
                      "quote_snapshots", "depth_snapshots", "options_chain"]:
            df = self.query(f"SELECT COUNT(*) AS n FROM {table}")
            out[table] = int(df["n"].iloc[0])
        return out


# ── pure adjustment logic (no DB needed, fully testable) ───────────────────

def adjust_ohlcv(df: pd.DataFrame, events: pd.DataFrame | None) -> pd.DataFrame:
    """Backward-adjust raw OHLCV bars for splits, bonuses and dividends.

    Adjustment factors are computed only from events with ``ex_date`` AFTER
    each bar — later events (relative to the bar) are the only ones that
    change a bar's current value, which is exactly what a backward-adjusted
    series must apply. Passing an events table already filtered to
    ``ex_date <= as_of`` guarantees no look-ahead bias.

    Exact total-return adjustment:
        CF(t)        = product of split/bonus ratios for events with ex_date > t
        d_scale(e)   = dividend(e) * product of split ratios of events AFTER e
        adj_price(t) = raw_price(t) * CF(t) - sum(d_scale(e) for ex_date(e) > t)
        adj_vol(t)   = raw_volume(t) * CF(t)

    Args:
        df: DataFrame indexed by date/ts with open/high/low/close/volume.
        events: DataFrame with columns ex_date, action_type, ratio,
                dividend_per_share (optional; empty frames tolerated).

    Returns:
        Adjusted DataFrame with the same index and columns.
    """
    result = df.copy()
    if events is None or events.empty:
        return result

    events = events.copy()
    events["ex_date"] = pd.to_datetime(events["ex_date"]).dt.normalize()
    if "ratio" not in events.columns:
        events["ratio"] = None
    if "dividend_per_share" not in events.columns:
        events["dividend_per_share"] = None
    events = events.sort_values("ex_date")

    split_events = events[events["action_type"].isin(["split", "bonus"])]
    div_events = events[events["action_type"] == "dividend"]

    split_ratio = {}
    for _, e in split_events.iterrows():
        ratio = float(e["ratio"]) if pd.notna(e["ratio"]) else 1.0
        if ratio and ratio != 1.0:
            split_ratio[e["ex_date"]] = ratio
    dividend_scale = {}
    for _, e in div_events.iterrows():
        d = float(e["dividend_per_share"]) if pd.notna(e["dividend_per_share"]) else 0.0
        if d > 0:
            dividend_scale[e["ex_date"]] = d

    if not split_ratio and not dividend_scale:
        return result

    # d_scale(e) = dividend(e) * product of split ratios of events AFTER e
    event_dates = sorted(set(split_ratio) | set(dividend_scale))
    n_events = len(event_dates)

    # suffix products of split ratios over all events (suffix[k] = product of ratios for events k..n-1)
    suffix = [1.0] * (n_events + 1)
    for k in range(n_events - 1, -1, -1):
        suffix[k] = suffix[k + 1] * split_ratio.get(event_dates[k], 1.0)

    # d_scale per event, then suffix sums over events (div_suffix[k] = sum of d_scale for events k..n-1)
    d_scale = [dividend_scale.get(d, 0.0) * suffix[k + 1] for k, d in enumerate(event_dates)]
    div_suffix = [0.0] * (n_events + 1)
    for k in range(n_events - 1, -1, -1):
        div_suffix[k] = div_suffix[k + 1] + d_scale[k]

    bar_dates = pd.to_datetime(result.index).normalize()
    event_arr = pd.to_datetime(event_dates)
    idx = event_arr.searchsorted(bar_dates, side="right")
    cf = pd.Series([suffix[i] for i in idx], index=result.index)
    div_sum = pd.Series([div_suffix[i] for i in idx], index=result.index)

    for col in ["open", "high", "low", "close"]:
        if col in result.columns:
            result[col] = result[col] * cf - div_sum
    if "volume" in result.columns:
        result["volume"] = result["volume"] * cf
    return result


def _parse_split_ratio(raw) -> float | None:
    """Parse a Yahoo Finance split value into a share multiplier.

    "5:1" -> 5.0, "1:2" -> 1.5 (bonus style), "3:2" -> 1.5.
    """
    try:
        if isinstance(raw, str) and ":" in raw:
            a, b = raw.split(":")
            a = float(a)
            b = float(b)
            if b == 0:
                return None
            return (a + b) / b if a < b else a / b
        val = float(raw)
        return val if val > 0 else None
    except (TypeError, ValueError):
        return None


def _df_to_rows(df: pd.DataFrame, symbol: str, source: str, daily: bool) -> list[tuple]:
    if df is None or df.empty:
        return []
    time_col = "date" if daily else "ts"
    work = df.copy()
    if work.index.name in ("date", "ts") or isinstance(work.index, pd.DatetimeIndex):
        work = work.reset_index()
        if work.columns[0] != time_col:
            work = work.rename(columns={work.columns[0]: time_col})
    cols = {c.lower(): c for c in work.columns}
    if time_col not in cols:
        raise ValueError(f"Expected a '{time_col}' column in the DataFrame")
    tcol = cols[time_col]
    rows = []
    for _, row in work.iterrows():
        ts_val = row[tcol]
        ts_str = pd.Timestamp(ts_val).isoformat() if pd.notna(ts_val) else None
        if ts_str is None:
            continue
        o = _f(row.get(cols.get("open", "")))
        h = _f(row.get(cols.get("high", "")))
        lo = _f(row.get(cols.get("low", "")))
        c = _f(row.get(cols.get("close", "")))
        v = _i(row.get(cols.get("volume", "")))
        rows.append((symbol, ts_str, o, h, lo, c, v, source))
    return rows


def _f(val) -> float | None:
    try:
        if val is None or pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _i(val) -> int | None:
    try:
        if val is None or pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


_store_singleton: MarketDataStore | None = None
_store_lock = threading.Lock()


def get_store(db_path: str | None = None) -> MarketDataStore:
    """Return a process-wide singleton store (avoids multiple DuckDB handles)."""
    global _store_singleton
    with _store_lock:
        if _store_singleton is None or (db_path and _store_singleton.db_path != db_path):
            if _store_singleton is not None:
                _store_singleton.close()
            _store_singleton = MarketDataStore(db_path or MARKET_DATA_DB)
        return _store_singleton

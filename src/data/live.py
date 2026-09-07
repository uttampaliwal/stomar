"""Free / low-latency live market data engine with a pluggable fetcher layer.

Architecture:
    IDataFetcher (ABC)
    ├── NSEFetcher       — NSE equity quotes + top-5 market depth, PCR, FII/DII
    ├── YahooFetcher     — Yahoo Finance chart HTTP polling (3-5s interval)
    ├── GoogleFetcher    — Google Finance quote page scrape (price fallback)
    └── KiteConnectFetcher — Zerodha Kite Connect WebSocket/HTTP (when keys provided)

    LiveDataEngine orchestrates a chain of fetchers: primary → fallbacks,
    caches quotes/depth in memory, aggregates 1-minute bars into the
    point-in-time store, and refreshes PCR / FII-DII snapshots.

Usage:
    from src.data.live import live_engine
    quote = live_engine.get_quote("RELIANCE.NS")
    depth = live_engine.get_depth("RELIANCE.NS")
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd
import requests

from src.core.settings import settings
from src.data.resilience import retry_with_backoff
from src.data.store import MarketDataStore, get_store
from src.data.universes import yf_ticker

logger = logging.getLogger(__name__)

DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

_NSE_HEADERS = {
    "User-Agent": DEFAULT_USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nseindia.com/",
}

# ─── data payloads ──────────────────────────────────────────────────────────


@dataclass
class DepthLevel:
    price: float
    quantity: int
    orders: int = 0


@dataclass
class LiveQuote:
    symbol: str
    price: float | None = None
    open: float | None = None
    high: float | None = None
    low: float | None = None
    prev_close: float | None = None
    change: float | None = None
    change_pct: float | None = None
    volume: int | None = None
    bid: float | None = None
    ask: float | None = None
    source: str = "unknown"
    ts: datetime = field(default_factory=datetime.now)
    is_stale: bool = False

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "price": self.price,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "prev_close": self.prev_close,
            "change": self.change,
            "change_pct": self.change_pct,
            "volume": self.volume,
            "bid": self.bid,
            "ask": self.ask,
            "source": self.source,
            "timestamp": self.ts.isoformat(),
            "is_stale": self.is_stale,
        }


@dataclass
class MarketDepth:
    symbol: str
    bids: list[DepthLevel] = field(default_factory=list)   # top of book first
    asks: list[DepthLevel] = field(default_factory=list)   # top of book first
    source: str = "unknown"
    ts: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bids": [{"price": b.price, "quantity": b.quantity, "orders": b.orders} for b in self.bids],
            "asks": [{"price": a.price, "quantity": a.quantity, "orders": a.orders} for a in self.asks],
            "source": self.source,
            "timestamp": self.ts.isoformat(),
        }


# ─── fetcher abstraction ────────────────────────────────────────────────────


class IDataFetcher(ABC):
    """Abstraction over any live market data provider (NSE, Yahoo, Google, Kite)."""

    name: str = "base"

    @abstractmethod
    def get_quote(self, symbol: str) -> LiveQuote:
        """Return the latest quote for a ticker. Raises on failure."""

    def get_depth(self, symbol: str) -> MarketDepth | None:
        """Return top-5 market depth, or None if the provider has no depth."""
        return None

    def get_pcr(self) -> dict | None:
        """Put-Call ratio for index options (NIFTY/BANKNIFTY) or None."""
        return None

    def get_fii_dii(self) -> dict | None:
        """Latest FII/DII net flows or None."""
        return None

    def is_available(self) -> bool:
        """Whether this provider can be used right now."""
        return True


def _symbol_to_nse(symbol: str) -> str:
    return symbol.upper().removesuffix(".NS")


# ─── NSE fetcher (primary) ──────────────────────────────────────────────────


class NSEFetcher(IDataFetcher):
    """NSE live quotes + top-5 market depth via nsepython with a raw-requests fallback.

    NSE requires session cookies; nsepython handles this when installed,
    otherwise we prime a requests session against the NSE homepage first.
    """

    name = "nse"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or self._build_session()
        self._nsepython = None
        try:
            import nsepython  # type: ignore

            self._nsepython = nsepython
            logger.info("NSEFetcher using nsepython %s", getattr(nsepython, "__version__", "?"))
        except ImportError:
            logger.info("nsepython not installed — NSEFetcher uses raw requests")

    @staticmethod
    def _build_session() -> requests.Session:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        try:
            session.get("https://www.nseindia.com", timeout=10)
        except Exception:
            pass
        return session

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def get_quote(self, symbol: str) -> LiveQuote:
        nse_symbol = _symbol_to_nse(symbol)
        if self._nsepython is not None:
            try:
                data = self._nsepython.nse_quote(nse_symbol)
            except TypeError:
                data = self._nsepython.quote(nse_symbol)
            return _parse_nse_quote(symbol, data, source="nse")
        data = self._raw_quote(nse_symbol)
        return _parse_nse_quote(symbol, data, source="nse")

    def get_depth(self, symbol: str) -> MarketDepth | None:
        nse_symbol = _symbol_to_nse(symbol)
        if self._nsepython is not None:
            try:
                data = self._nsepython.nse_quote(nse_symbol)
            except TypeError:
                data = self._nsepython.quote(nse_symbol)
        else:
            data = self._raw_quote(nse_symbol)
        return _parse_nse_depth(symbol, data)

    def _raw_quote(self, nse_symbol: str) -> dict:
        resp = self.session.get(
            f"https://www.nseindia.com/api/quote-equity?symbol={nse_symbol}",
            timeout=10,
        )
        resp.raise_for_status()
        return resp.json()

    def get_pcr(self) -> dict | None:
        if self._nsepython is not None:
            try:
                pcr = self._nsepython.pcr()
                if isinstance(pcr, dict) and pcr:
                    pcr["source"] = "nse"
                    return pcr
            except Exception as exc:
                logger.debug("nsepython PCR failed: %s", exc)
        from src.signals.flow import fetch_options_pcr

        pcr = fetch_options_pcr()
        return pcr if pcr else None

    def get_fii_dii(self) -> dict | None:
        if self._nsepython is not None:
            try:
                data = self._nsepython.nse_fiidii()
                if isinstance(data, dict) and data.get("data"):
                    latest = data["data"][0]
                    return {
                        "fii_net": _to_float(latest.get("netValue")),
                        "dii_net": 0.0,
                        "date": str(latest.get("date", ""))[:10],
                        "source": "nse",
                    }
            except Exception as exc:
                logger.debug("nsepython FII/DII failed: %s", exc)
        from src.signals.flow import fetch_fii_dii

        df = fetch_fii_dii()
        if df is None or df.empty:
            return None
        latest = df.iloc[-1]
        return {
            "fii_net": _to_float(latest.get("fii_net")),
            "dii_net": _to_float(latest.get("dii_net")),
            "date": str(pd.Timestamp(latest["date"]).date()),
            "source": "nse",
        }

    def is_available(self) -> bool:
        try:
            return self._raw_quote("NIFTY") is not None
        except Exception:
            return False


def _parse_nse_quote(symbol: str, data: dict, source: str = "nse") -> LiveQuote:
    info = data.get("priceInfo", {})
    ohlc = data.get("ohlc", {})
    last = _to_float(info.get("lastPrice")) or _to_float(data.get("lastPrice"))
    prev_close = _to_float(info.get("previousClose")) or _to_float(data.get("previousClose"))
    change = _to_float(info.get("change")) or _to_float(data.get("change"))
    change_pct = _to_float(info.get("pChange")) or _to_float(data.get("pChange"))
    open_ = _to_float(ohlc.get("open"))
    high = _to_float(ohlc.get("high"))
    low = _to_float(ohlc.get("low"))
    volume = _to_int(data.get("totalTradedVolume")) or _to_int(info.get("totalTradedVolume"))
    return LiveQuote(
        symbol=symbol, price=last, open=open_, high=high, low=low,
        prev_close=prev_close, change=change, change_pct=change_pct,
        volume=volume, source=source,
    )


def _parse_nse_depth(symbol: str, data: dict) -> MarketDepth:
    book = data.get("marketDeptOrderBook", {})
    bids = [
        DepthLevel(_to_float(level.get("price")), _to_int(level.get("quantity")) or 0,
                   _to_int(level.get("numberOfOrders")) or 0)
        for level in book.get("bid", [])[:5]
        if _to_float(level.get("price")) is not None
    ]
    asks = [
        DepthLevel(_to_float(level.get("price")), _to_int(level.get("quantity")) or 0,
                   _to_int(level.get("numberOfOrders")) or 0)
        for level in book.get("ask", [])[:5]
        if _to_float(level.get("price")) is not None
    ]
    if not bids and not asks:
        raise ValueError(f"No depth data for {symbol}")
    return MarketDepth(symbol=symbol, bids=bids, asks=asks, source="nse")


# ─── Yahoo Finance fetcher (fallback, fast polling) ─────────────────────────


class YahooFetcher(IDataFetcher):
    """Yahoo Finance chart endpoint — lightweight HTTP polling (3-5s)."""

    name = "yahoo"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    @retry_with_backoff(max_retries=2, base_delay=1.0, max_delay=6.0)
    def get_quote(self, symbol: str) -> LiveQuote:
        ticker = yf_ticker(symbol)
        resp = self.session.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}",
            params={"interval": "1m", "range": "1d", "includePrePost": "false"},
            timeout=8,
        )
        resp.raise_for_status()
        data = resp.json()
        result = data["chart"]["result"][0]
        meta = result["meta"]
        price = _to_float(meta.get("regularMarketPrice"))
        prev_close = _to_float(meta.get("chartPreviousClose")) or _to_float(meta.get("previousClose"))
        change = None
        change_pct = None
        if price is not None and prev_close:
            change = round(price - prev_close, 4)
            change_pct = round(change / prev_close * 100, 4)
        timestamps = result.get("timestamp") or []
        quote = (result.get("indicators", {}).get("quote") or [{}])[0]
        volume = None
        if timestamps:
            volumes = quote.get("volume") or []
            volume = _to_int(volumes[-1]) if volumes else None
        return LiveQuote(
            symbol=symbol, price=price, prev_close=prev_close,
            change=change, change_pct=change_pct, volume=volume,
            source="yahoo", ts=datetime.now(),
        )

    def is_available(self) -> bool:
        return True


# ─── Google Finance fetcher (last-resort price fallback) ────────────────────


class GoogleFetcher(IDataFetcher):
    """Google Finance quote page scrape — price-only last-resort fallback."""

    name = "google"

    def __init__(self, session: requests.Session | None = None):
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": DEFAULT_USER_AGENT})

    @retry_with_backoff(max_retries=1, base_delay=0.8, max_delay=3.0)
    def get_quote(self, symbol: str) -> LiveQuote:
        exchange = "NSE" if symbol.endswith(".NS") else "BSE"
        nse_symbol = _symbol_to_nse(symbol)
        resp = self.session.get(
            f"https://www.google.com/finance/quote/{nse_symbol}:{exchange}",
            timeout=8,
        )
        resp.raise_for_status()
        price = _extract_gf_price(resp.text, nse_symbol, exchange)
        if price is None:
            raise ValueError(f"No price found on Google Finance for {symbol}")
        return LiveQuote(symbol=symbol, price=price, source="google")

    def is_available(self) -> bool:
        return True


def _extract_gf_price(html: str, symbol: str, exchange: str) -> float | None:
    """Pull the current price from Google Finance's embedded JSON.

    Google Finance embeds price data as ``[1, <price>, 1, "<price>", 0, 0,
    null, 1]`` arrays near the ``"SYMBOL:EXCHANGE"`` marker. This is a
    best-effort fallback parser for the HTML page.
    """
    marker = f'"{symbol}:{exchange}"'
    idx = html.find(marker)
    if idx == -1:
        return None
    window = html[idx: idx + 800]
    import re

    # primary: the [1, price, 1, "price"] current-price array
    for pat in [
        r'\[1,\s*(\d+(?:\.\d+)?),\s*1,\s*"[0-9,.]+"',
        r'\[1,\s*(\d+(?:\.\d+)?)\s*,',
    ]:
        hit = re.search(pat, window)
        if hit:
            raw = hit.group(1)
            try:
                return float(raw)
            except ValueError:
                continue
    # last resort: any plausible number near the marker
    for hit in re.finditer(r'"fmt":"([\d.,]+)"', window[:600]):
        cleaned = hit.group(1).replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return None


# ─── Zerodha Kite Connect fetcher (plug-in when API key provided) ───────────


class KiteConnectFetcher(IDataFetcher):
    """Zerodha Kite Connect — enabled only when credentials are configured.

    Reads ``STOMAR_KITE_API_KEY`` and ``STOMAR_KITE_ACCESS_TOKEN``. All
    imports of the ``kiteconnect`` package are lazy, so this module stays
    importable without the SDK installed.
    """

    name = "kite"

    def __init__(self, api_key: str | None = None, access_token: str | None = None):
        self.api_key = api_key or settings.kite_api_key
        self.access_token = access_token or settings.kite_access_token
        if not self.api_key or not self.access_token:
            raise ValueError("KiteConnectFetcher requires STOMAR_KITE_API_KEY and STOMAR_KITE_ACCESS_TOKEN")
        try:
            from kiteconnect import KiteConnect  # type: ignore

            self._kite = KiteConnect(api_key=self.api_key)
            self._kite.set_access_token(self.access_token)
        except ImportError as exc:
            raise ImportError(
                "kiteconnect package not installed — `pip install kiteconnect` "
                "to use KiteConnectFetcher"
            ) from exc

    @retry_with_backoff(max_retries=2, base_delay=0.5)
    def get_quote(self, symbol: str) -> LiveQuote:
        instrument = _kite_instrument(symbol)
        data = self._kite.quote([instrument])[instrument]
        ohlc = data.get("ohlc", {})
        depth = (data.get("depth", {}) or {})
        return LiveQuote(
            symbol=symbol,
            price=_to_float(data.get("last_price")),
            open=_to_float(ohlc.get("open")),
            high=_to_float(ohlc.get("high")),
            low=_to_float(ohlc.get("low")),
            prev_close=_to_float(ohlc.get("close")),
            volume=_to_int(data.get("volume")),
            bid=_to_float(data.get("last_price")),
            ask=_to_float(data.get("last_price")),
            source="kite",
        )

    def get_depth(self, symbol: str) -> MarketDepth | None:
        instrument = _kite_instrument(symbol)
        data = self._kite.quote([instrument])[instrument]
        depth = data.get("depth", {}) or {}
        bids = [
            DepthLevel(_to_float(level.get("price")), _to_int(level.get("quantity")) or 0,
                       _to_int(level.get("orders")) or 0)
            for level in depth.get("buy", [])[:5]
        ]
        asks = [
            DepthLevel(_to_float(level.get("price")), _to_int(level.get("quantity")) or 0,
                       _to_int(level.get("orders")) or 0)
            for level in depth.get("sell", [])[:5]
        ]
        return MarketDepth(symbol=symbol, bids=bids, asks=asks, source="kite")

    def is_available(self) -> bool:
        return True


def _kite_instrument(symbol: str) -> str:
    return f"NSE:{_symbol_to_nse(symbol)}"


# ─── live engine ────────────────────────────────────────────────────────────


class LiveDataEngine:
    """Orchestrates fetchers with caching, fallback and minute-bar aggregation."""

    QUOTE_TTL = 3.0
    DEPTH_TTL = 5.0

    def __init__(
        self,
        fetchers: list[IDataFetcher] | None = None,
        store: MarketDataStore | None = None,
        poll_interval: float | None = None,
        poll_symbols: list[str] | None = None,
    ):
        self.store = store or get_store()
        self.poll_interval = poll_interval or settings.live_poll_interval
        self.poll_symbols = poll_symbols or settings.live_universe
        self.fetchers = fetchers or self._default_fetchers()
        self._quotes: dict[str, tuple[float, LiveQuote]] = {}
        self._depths: dict[str, tuple[float, MarketDepth]] = {}
        self._pcr: tuple[float, dict] | None = None
        self._fii_dii: tuple[float, dict] | None = None
        self._minute_accum: dict[str, list[float]] = {}  # symbol -> [open, high, low, close, vol]
        self._minute_key: dict[str, datetime] = {}
        self._last_volume: dict[str, int] = {}  # symbol -> last observed cumulative volume
        self._lock = threading.RLock()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._last_health: dict[str, bool] = {}

    @staticmethod
    def _default_fetchers() -> list[IDataFetcher]:
        fetchers: list[IDataFetcher] = [NSEFetcher(), YahooFetcher()]
        try:
            fetchers.insert(0, KiteConnectFetcher())
            logger.info("Kite Connect fetcher enabled (credentials provided)")
        except (ImportError, ValueError) as exc:
            logger.info("Kite Connect fetcher disabled: %s", exc)
        fetchers.append(GoogleFetcher())
        return fetchers

    # ── quote / depth with fallback chain ────────────────────────────────

    def get_quote(self, symbol: str, force: bool = False) -> LiveQuote:
        now = time.monotonic()
        with self._lock:
            cached = self._quotes.get(symbol)
            if cached and not force and now - cached[0] < self.QUOTE_TTL:
                return cached[1]

        errors = []
        for fetcher in self.fetchers:
            if not fetcher.is_available():
                continue
            try:
                quote = fetcher.get_quote(symbol)
                from src.core.timeutils import now_ist
                quote.ts = now_ist()
                with self._lock:
                    self._quotes[symbol] = (time.monotonic(), quote)
                try:
                    self.store.record_quote(
                        symbol, quote.price or 0.0,
                        change_pct=quote.change_pct or 0.0,
                        volume=quote.volume or 0,
                        bid=quote.bid, ask=quote.ask, source=quote.source,
                    )
                except Exception as exc:
                    logger.debug("quote snapshot write failed: %s", exc)
                self._mark_health(fetcher, True)
                self._accumulate_minute(symbol, quote)
                return quote
            except Exception as exc:
                errors.append(f"{fetcher.name}: {exc}")
                self._mark_health(fetcher, False)
                logger.debug("quote fallback for %s (%s)", symbol, errors[-1])
                continue

        with self._lock:
            stale = self._quotes.get(symbol)
            if stale and stale[1].price is not None:
                stale_quote = stale[1]
                stale_quote.is_stale = True
                return stale_quote
        raise RuntimeError(f"All fetchers failed for {symbol}: {'; '.join(errors)}")

    def get_depth(self, symbol: str, force: bool = False) -> MarketDepth:
        now = time.monotonic()
        with self._lock:
            cached = self._depths.get(symbol)
            if cached and not force and now - cached[0] < self.DEPTH_TTL:
                return cached[1]

        errors = []
        for fetcher in self.fetchers:
            try:
                depth = fetcher.get_depth(symbol)
                if depth is None:
                    continue
                depth.ts = datetime.now()
                with self._lock:
                    self._depths[symbol] = (time.monotonic(), depth)
                try:
                    self.store.record_depth(symbol, depth.to_dict(), source=depth.source)
                except Exception as exc:
                    logger.debug("depth snapshot write failed: %s", exc)
                return depth
            except Exception as exc:
                errors.append(f"{fetcher.name}: {exc}")
                continue
        raise RuntimeError(f"No depth source for {symbol}: {'; '.join(errors)}")

    # ── PCR / FII-DII with TTL ───────────────────────────────────────────

    def get_pcr(self, force: bool = False) -> dict | None:
        now = time.monotonic()
        with self._lock:
            if self._pcr and not force and now - self._pcr[0] < 300:
                return self._pcr[1]
        for fetcher in self.fetchers:
            try:
                pcr = fetcher.get_pcr()
                if pcr:
                    with self._lock:
                        self._pcr = (time.monotonic(), pcr)
                    return pcr
            except Exception:
                continue
        return None

    def get_fii_dii(self, force: bool = False) -> dict | None:
        now = time.monotonic()
        with self._lock:
            if self._fii_dii and not force and now - self._fii_dii[0] < 600:
                return self._fii_dii[1]
        for fetcher in self.fetchers:
            try:
                flows = fetcher.get_fii_dii()
                if flows:
                    with self._lock:
                        self._fii_dii = (time.monotonic(), flows)
                    return flows
            except Exception:
                continue
        return None

    # ── minute-bar aggregation ───────────────────────────────────────────

    def _accumulate_minute(self, symbol: str, quote: LiveQuote):
        price = quote.price
        if price is None:
            return
        now = datetime.now()
        key = now.replace(second=0, microsecond=0)
        volume = int(quote.volume or 0)
        with self._lock:
            # Quote volume is the session-cumulative traded volume, so only
            # the delta since the previous observation belongs to this bar.
            # Adding the raw value on every poll would multiply the daily
            # volume by the number of polls per minute (#24).
            last_vol = self._last_volume.get(symbol)
            if last_vol is None:
                delta = 0
            else:
                delta = max(0, volume - last_vol)  # new session: reset to 0
            self._last_volume[symbol] = volume

            current_key = self._minute_key.get(symbol)
            if current_key != key:
                self._flush_minute(symbol)
                self._minute_key[symbol] = key
                self._minute_accum[symbol] = [price, price, price, price, 0]
            else:
                bar = self._minute_accum[symbol]
                bar[1] = max(bar[1], price)
                bar[2] = min(bar[2], price)
                bar[3] = price
            self._minute_accum[symbol][4] += delta

    def _flush_minute(self, symbol: str):
        bar = self._minute_accum.pop(symbol, None)
        key = self._minute_key.pop(symbol, None)
        if bar is None or key is None:
            return
        try:
            self.store.upsert_minute_bars(
                symbol,
                pd.DataFrame([{
                    "ts": key, "open": bar[0], "high": bar[1],
                    "low": bar[2], "close": bar[3], "volume": bar[4],
                }]),
                source="live_engine",
            )
        except Exception as exc:
            logger.debug("minute bar write failed: %s", exc)

    def flush_all_minutes(self):
        with self._lock:
            for symbol in list(self._minute_accum):
                self._flush_minute(symbol)

    # ── background poller ────────────────────────────────────────────────

    def start(self, daemon: bool = True):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="live-engine", daemon=daemon)
        self._thread.start()
        logger.info("Live engine started (interval=%.1fs, %d symbols)",
                    self.poll_interval, len(self.poll_symbols))

    def stop(self):
        self._stop.set()
        self.flush_all_minutes()
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Live engine stopped")

    def _poll_loop(self):
        while not self._stop.is_set():
            started = time.monotonic()
            for symbol in self.poll_symbols:
                if self._stop.is_set():
                    break
                try:
                    self.get_quote(symbol)
                except Exception as exc:
                    logger.debug("poll %s failed: %s", symbol, exc)
            # Optional slow-path refreshes
            if datetime.now().minute % 5 == 0:
                try:
                    self.get_pcr()
                    self.get_fii_dii()
                except Exception:
                    pass
            elapsed = time.monotonic() - started
            remaining = self.poll_interval - elapsed
            if remaining > 0:
                self._stop.wait(timeout=min(remaining, 5.0))

    def _mark_health(self, fetcher: IDataFetcher, ok: bool):
        with self._lock:
            self._last_health[fetcher.name] = ok

    def health(self) -> dict:
        with self._lock:
            return {
                "running": bool(self._thread and self._thread.is_alive()),
                "fetchers": dict(self._last_health),
                "cached_quotes": len(self._quotes),
                "cached_depths": len(self._depths),
            }


def _to_float(val) -> float | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        return float(val)
    except (TypeError, ValueError):
        return None


def _to_int(val) -> int | None:
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
        return int(float(val))
    except (TypeError, ValueError):
        return None


_live_engine: LiveDataEngine | None = None
_engine_lock = threading.Lock()


def get_live_engine() -> LiveDataEngine:
    """Process-wide live engine singleton."""
    global _live_engine
    with _engine_lock:
        if _live_engine is None:
            _live_engine = LiveDataEngine()
        return _live_engine


live_engine = get_live_engine()

"""Stock universe definitions: NIFTY 50, NIFTY NEXT 50, NIFTY MIDCAP 100 + custom.

Symbols are stored in NSE format (e.g. "RELIANCE") and exposed as both
NSE symbols (for nsepython / NSE APIs) and Yahoo Finance tickers
(e.g. "RELIANCE.NS") for yfinance / Yahoo chart polling.

Example:
    from src.data.universes import get_universe, yf_ticker
    universe = get_universe("nifty_50")          # NSE symbols
    yf_symbols = [yf_ticker(s) for s in universe]  # "RELIANCE.NS"
"""

from __future__ import annotations

import logging
from typing import Iterable

logger = logging.getLogger(__name__)

# ─── NIFTY 50 (as of 2026-07 snapshot; refresh via refresh_universe_from_nse) ───
NIFTY_50 = [
    "RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR", "ITC",
    "SBIN", "BHARTIARTL", "KOTAKBANK", "BAJFINANCE", "LT", "WIPRO", "AXISBANK",
    "TITAN", "MARUTI", "SUNPHARMA", "ASIANPAINT", "NTPC", "ONGC", "M&M",
    "ULTRACEMCO", "ADANIENT", "ADANIPORTS", "BAJAJFINSV", "HCLTECH", "TECHM",
    "NESTLEIND", "POWERGRID", "TATAMOTORS", "TATASTEEL", "JSWSTEEL", "GRASIM",
    "DRREDDY", "CIPLA", "APOLLOHOSP", "INDUSINDBK", "HDFCLIFE", "SBILIFE",
    "BRITANNIA", "EICHERMOT", "COALINDIA", "TATACONSUM", "DIVISLAB", "BAJAJ-AUTO",
    "HEROMOTOCO", "TATAPOWER", "HINDALCO", "LTIM", "BPCL",
]

# ─── NIFTY NEXT 50 ───
NIFTY_NEXT_50 = [
    "BANKBARODA",
    "IDEA",
    "YESBANK",
    "PFC",
    "RECLTD",
    "VEDL",
    "INDUSTOWER",
    "ZOMATO",
    "IRFC",
    "DMART",
    "DLF",
    "PIDILITIND",
    "SIEMENS",
    "BOSCHLTD",
    "HAVELLS",
    "ASHOKLEY",
    "AMBUJACEM",
    "ACC",
    "GAIL",
    "HAL",
    "BEL",
    "HINDZINC",
    "DABUR",
    "MARICO",
    "COLGATE",
    "GODREJCP",
    "ICICIPRULI",
    "ICICIGI",
    "SRTRANSFIN",
    "MUTHOOTFIN",
    "CHOLAFIN",
    "BANDHANBNK",
    "IOC",
    "SHREECEM",
    "DIXON",
    "POLYCAB",
    "TRENT",
    "JINDALSTEL",
    "NAUKRI",
    "TVSMOTOR",
    "MCDOWELL-N",
    "PAGEIND",
    "SYNGENE",
    "PERSISTENT",
    "AUBANK",
    "MOTHERSON",
    "CONCOR",
    "ESCORTS",
    "CANBK",
    "ABB",
]

# ─── NIFTY MIDCAP 100 ───
NIFTY_MIDCAP_100 = [
    "AARTIIND",
    "ABB",
    "ABFRL",
    "ALKEM",
    "APLAPOLLO",
    "AUROPHARMA",
    "BALKRISIND",
    "BATAINDIA",
    "BERGEPAINT",
    "BIOCON",
    "BSE",
    "CGPOWER",
    "CROMPTON",
    "CUB",
    "CUMMINSIND",
    "CYIENT",
    "DELHIVERY",
    "DEVYANI",
    "EIDPARRY",
    "ENDURANCE",
    "ENGINERSIN",
    "EXIDEIND",
    "FEDERALBNK",
    "FINPIPE",
    "FORTIS",
    "GODREJPROP",
    "GODREJIND",
    "GMRINFRA",
    "GODFRYPHLP",
    "GRINDWELL",
    "GSFC",
    "GUJGASLTD",
    "HAPPSTMNDS",
    "HINDCOPPER",
    "HINDPETRO",
    "HUDCO",
    "IDBI",
    "IDFC",
    "IDFCFIRSTB",
    "INDHOTEL",
    "INDIGO",
    "IRB",
    "IRCTC",
    "JBCHEPHARM",
    "JINDALSAW",
    "JIOFIN",
    "JKCEMENT",
    "JSL",
    "JUBLFOOD",
    "KALPATPOWR",
    "KPITTECH",
    "KSCL",
    "LICHSGFIN",
    "LODHA",
    "LUPIN",
    "MANYAVAR",
    "MAXHEALTH",
    "METROPOLIS",
    "MGL",
    "MINDTREE",
    "NBCC",
    "NCC",
    "NHPC",
    "NUVOCO",
    "OIL",
    "PATANJALI",
    "PAYTM",
    "PEL",
    "PHOENIXLTD",
    "PIIND",
    "PNB",
    "POWERINDIA",
    "PRESTIGE",
    "RBLBANK",
    "RATNAMANI",
    "REDINGTON",
    "RVNL",
    "SAIL",
    "SANOFI",
    "SHRIRAMFIN",
    "SKFINDIA",
    "SONACOMS",
    "SUNDARMFIN",
    "SUNDRMFAST",
    "SUPREMEIND",
    "SWSOLAR",
    "SYRMA",
    "TATACHEM",
    "TATACOMM",
    "TATAMETALI",
    "TIINDIA",
    "TORNTPHARM",
    "TRIDENT",
    "UPL",
    "UNIONBANK",
    "VBL",
    "VOLTAS",
    "WESTLIFE",
    "ZYDUSLIFE",
    "KAYNES",
]

# ─── Index tickers (Yahoo Finance symbols) ───
INDEX_TICKERS = {
    "nifty_50": "^NSEI",
    "nifty_next_50": "NIFTYNEXT50.NS",
    "bank_nifty": "^NSEBANK",
}

# NSE index names → Yahoo symbols
NSE_INDEX_MAP = {
    "NIFTY 50": "^NSEI",
    "NIFTY NEXT 50": "NIFTYNEXT50.NS",
    "NIFTY BANK": "^NSEBANK",
    "NIFTY MIDCAP 100": "NIFTYMIDCAP100.NS",
}

UNIVERSES: dict[str, list[str]] = {
    "nifty_50": NIFTY_50,
    "nifty_next_50": NIFTY_NEXT_50,
    "nifty_midcap_100": NIFTY_MIDCAP_100,
}

UNIVERSE_ALIASES = {
    "nifty50": "nifty_50",
    "nifty": "nifty_50",
    "next50": "nifty_next_50",
    "niftynext50": "nifty_next_50",
    "midcap100": "nifty_midcap_100",
    "midcap": "nifty_midcap_100",
    "niftymidcap100": "nifty_midcap_100",
}


def nse_symbols(universe: Iterable[str]) -> list[str]:
    """Normalize any symbol iterable to plain NSE symbols (strip .NS)."""
    return [s.upper().removesuffix(".NS") for s in universe]


def yf_ticker(symbol: str) -> str:
    """Convert an NSE symbol to a Yahoo Finance ticker (append .NS)."""
    symbol = symbol.upper()
    if symbol.startswith("^"):
        return symbol
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def yf_tickers(universe: Iterable[str]) -> list[str]:
    """Convert an NSE symbol list to Yahoo Finance tickers."""
    return [yf_ticker(s) for s in universe]


def get_universe(name: str, as_yf: bool = False) -> list[str]:
    """Return a stock universe by name.

    Args:
        name: "nifty_50" | "nifty_next_50" | "nifty_midcap_100" or any alias.
        as_yf: If True, return Yahoo Finance tickers (.NS suffixed).
    """
    key = UNIVERSE_ALIASES.get(name.lower(), name.lower())
    if key not in UNIVERSES:
        raise KeyError(
            f"Unknown universe '{name}'. Available: {sorted(UNIVERSES)}"
        )
    symbols = list(UNIVERSES[key])
    return yf_tickers(symbols) if as_yf else symbols


def all_universes(as_yf: bool = False) -> list[str]:
    """Concatenated symbols of all predefined universes (deduplicated)."""
    seen: set[str] = set()
    result: list[str] = []
    for symbols in UNIVERSES.values():
        for s in symbols:
            if s not in seen:
                seen.add(s)
                result.append(s)
    return yf_tickers(result) if as_yf else result


def custom_universe(symbols: Iterable[str], as_yf: bool = False) -> list[str]:
    """Build a custom universe from an arbitrary symbol list.

    Accepts NSE symbols, .NS suffixed tickers, or Yahoo symbols.
    """
    result = nse_symbols(symbols)
    return yf_tickers(result) if as_yf else result


def refresh_universe_from_nse(name: str = "nifty_50") -> list[str]:
    """Fetch the current index constituents from NSE's public API.

    Best-effort: falls back to the bundled snapshot on failure.

    Args:
        name: Index to fetch — "nifty_50" | "nifty_next_50" | "nifty_midcap_100".

    Returns:
        The (possibly refreshed) universe as NSE symbols.
    """
    index_name = {
        "nifty_50": "NIFTY 50",
        "nifty_next_50": "NIFTY NEXT 50",
        "nifty_midcap_100": "NIFTY MIDCAP 100",
    }.get(name.lower(), name)
    try:
        import requests

        resp = requests.get(
            f"https://www.nseindia.com/api/equity-stockIndices?index={index_name}",
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            },
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        symbols = [
            row["symbol"] for row in data.get("data", []) if row.get("symbol")
        ]
        if not symbols:
            raise ValueError(f"No constituents returned for {index_name}")
        UNIVERSES[name] = list(dict.fromkeys(symbols))
        logger.info("Refreshed universe %s from NSE (%d symbols)", name, len(symbols))
        return symbols
    except Exception as exc:
        logger.warning("NSE universe refresh failed (%s) — using snapshot", exc)
        return get_universe(name)

"""Multi-source data fetching with automatic fallback.

Primary: yfinance
Fallback: NSE India archive (best-effort free)

Usage:
    result = fetch_with_fallback("RELIANCE.NS", period="2y")
    df = result["df"]
    source = result["source"]
    validation = result["validation"]
"""

import logging
from abc import ABC, abstractmethod
from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.data_validation import validate_data
from src.data.resilience import retry_with_backoff, yf_breaker, nse_breaker

logger = logging.getLogger(__name__)


class DataSource(ABC):
    """Base class for data sources."""

    @abstractmethod
    def fetch(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        """Fetch price data for a ticker."""

    def validate(self, df: pd.DataFrame) -> bool:
        """Quick sanity check on fetched data."""
        return df is not None and len(df) > 10 and not df.empty


class YFinanceSource(DataSource):
    """Primary source: yfinance."""

    @retry_with_backoff(max_retries=2, base_delay=1.0)
    def fetch(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        stock = yf.Ticker(ticker)
        df = yf_breaker.call(stock.history, period=period, interval=interval)

        if df.empty:
            raise ValueError(f"yfinance returned empty data for {ticker}")

        df.columns = [c.lower() for c in df.columns]
        df.index = pd.to_datetime(df.index)
        df.index.name = "date"
        return df

    def validate(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        required = ["open", "high", "low", "close", "volume"]
        has_cols = all(c in df.columns for c in required)
        return has_cols and len(df) > 10


class NSEArchiveSource(DataSource):
    """Fallback: NSE India archive.

    Attempts to fetch from NSE's public CSV archives.
    This is best-effort and may break if NSE changes their website.
    """

    @retry_with_backoff(max_retries=2, base_delay=2.0)
    def fetch(self, ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
        import requests

        symbol = ticker.replace(".NS", "")
        now = datetime.now()
        url = f"https://archives.nseindia.com/content/historical/EQUITIES/{now.year}/cm{symbol}{now.strftime('%b%Y')}bhav.csv.zip"

        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0",
            "Accept": "text/csv",
        })

        try:
            resp = nse_breaker.call(session.get, url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            raise ValueError(f"NSE archive fetch failed for {symbol}: {e}")

        from io import BytesIO
        import zipfile

        try:
            z = zipfile.ZipFile(BytesIO(resp.content))
            namelist = z.namelist()
            if not namelist:
                raise ValueError(f"Empty zip archive for {symbol}")
            csv_name = namelist[0]
            df = pd.read_csv(z.open(csv_name))
        except Exception as e:
            raise ValueError(f"Failed to parse NSE archive for {symbol}: {e}")

        # Standardize column names
        col_map = {
            "Date": "date", "Open": "open", "High": "high",
            "Low": "low", "Close": "close", "Volume": "volume",
        }
        df = df.rename(columns=col_map)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date").sort_index()
        df.index.name = "date"

        return df

    def validate(self, df: pd.DataFrame) -> bool:
        if df is None or df.empty:
            return False
        required = ["open", "high", "low", "close"]
        has_cols = all(c in df.columns for c in required)
        return has_cols and len(df) > 10


def fetch_with_fallback(
    ticker: str,
    period: str = "2y",
    interval: str = "1d",
    sources: list = None,
) -> dict:
    """Try primary source, fall back to secondary if it fails.

    Args:
        ticker: Stock ticker (e.g., "RELIANCE.NS")
        period: Data period (e.g., "2y")
        interval: Data interval (e.g., "1d")
        sources: List of DataSource instances to try (default: [YFinance, NSE])

    Returns:
        Dict with df, source, validation, timestamp

    Raises:
        ValueError: If all sources fail
    """
    if sources is None:
        sources = [YFinanceSource(), NSEArchiveSource()]

    last_error = None
    for source in sources:
        try:
            df = source.fetch(ticker, period, interval)
            if not source.validate(df):
                logger.warning(f"{source.__class__.__name__}: validation failed for {ticker}")
                continue

            validation = validate_data(df, ticker)

            logger.info(f"Fetched {ticker} from {source.__class__.__name__}: "
                        f"{len(df)} rows ({validation['date_range']})")

            return {
                "df": df,
                "source": source.__class__.__name__,
                "validation": validation,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            last_error = e
            logger.warning(f"{source.__class__.__name__} failed for {ticker}: {e}")
            continue

    raise ValueError(f"All data sources failed for {ticker}. Last error: {last_error}")

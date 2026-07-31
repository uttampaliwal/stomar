"""Free real-time market data endpoints: candles, quote, returns, sources."""

from fastapi import APIRouter, HTTPException, Query

from services.market_data import SUPPORTED_INTERVALS, market_data_service

router = APIRouter()


@router.get("/candles")
def candles(
    symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS"),
    interval: str = Query("15m", description="Bar interval: 1m, 5m, 15m or 1d"),
    limit: int = Query(500, ge=1, le=20000, description="Max bars to return"),
    indicators: bool = Query(True, description="Attach RSI/MACD/ATR/BB/Volume-Z columns"),
):
    """OHLCV candles with indicators for chart rendering.

    Sources are tried in fallback order: yfinance → local store → NSE archives
    → parquet cache. The payload's ``source`` field reports which feed served.
    """
    if interval not in SUPPORTED_INTERVALS:
        raise HTTPException(
            status_code=422,
            detail=f"interval must be one of {SUPPORTED_INTERVALS}, got {interval!r}",
        )
    try:
        return market_data_service.get_candles(
            symbol=symbol, interval=interval, limit=limit, indicators=indicators
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/quote")
def quote(symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS")):
    """Current price / change / volume with NSE → Yahoo → Google → Kite fallback."""
    try:
        return market_data_service.get_quote(symbol)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/returns")
def daily_returns(
    symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS"),
    days: int = Query(30, ge=2, le=20000, description="Number of sessions"),
):
    """Daily close-to-close returns for the last *days* sessions."""
    try:
        return market_data_service.get_daily_returns(symbol, days=days)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/sources")
def sources():
    """Fallback chain health: which providers serve which intervals."""
    return market_data_service.sources_health()

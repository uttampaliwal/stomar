"""Market status and pulse endpoint."""

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.data.data_fetcher import NSE_STOCKS, get_market_status
from src.data.store import get_store
from src.signals.flow import fetch_fii_dii, fetch_options_pcr, get_flow_sentiment
from src.signals.multitimeframe import fetch_mtf_data, get_combined_signal

router = APIRouter()


@router.get("/status")
def market_status():
    return {"status": get_market_status()}


@router.get("/stocks")
def list_stocks():
    return {"stocks": NSE_STOCKS}


@router.get("/history")
def market_history(
    symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS"),
    start: str | None = Query(None, description="Start date (YYYY-MM-DD)"),
    end: str | None = Query(None, description="End date (YYYY-MM-DD)"),
    as_of: str | None = Query(None, description="Point-in-time cutoff date — only corporate actions known by this date are applied"),
    adjusted: bool = Query(True, description="Apply backward split/bonus/dividend adjustment"),
    interval: str = Query("daily", pattern="^(daily|minute)$", description="Bar interval"),
    limit: int = Query(1000, ge=1, le=20000, description="Max rows to return"),
):
    """Point-in-time OHLCV history from the local store.

    Raw bars plus (optionally) backward-adjusted open/high/low/close/volume.
    With ``as_of`` set, later corporate actions are invisible — no look-ahead.
    """
    try:
        store = get_store()
        df = store.get_history(
            symbol=symbol,
            start=start,
            end=end,
            as_of=as_of,
            adjusted=adjusted,
            interval=interval,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if df.empty:
        return {"symbol": symbol, "interval": interval, "rows": 0, "bars": [], "adjusted": adjusted, "as_of": as_of}

    df = df.tail(limit)
    payload = []
    for idx, row in df.iterrows():
        bar = {"timestamp": str(idx), "date": str(idx)}
        if "source" in row:
            bar["source"] = row["source"]
        for col in ["open", "high", "low", "close", "volume"]:
            if col in row and pd.notna(row[col]):
                bar[col] = float(row[col]) if col != "volume" else int(row[col])
        if adjusted:
            for col in ["adjusted_open", "adjusted_high", "adjusted_low", "adjusted_close", "adjusted_volume"]:
                if col in row and pd.notna(row[col]):
                    bar[col] = float(row[col]) if col != "adjusted_volume" else int(row[col])
        payload.append(bar)

    return {
        "symbol": symbol,
        "interval": interval,
        "rows": len(payload),
        "adjusted": adjusted,
        "as_of": as_of,
        "bars": payload,
    }


@router.get("/live-quote")
def live_quote(symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS")):
    """Latest quote with automatic fallback across NSE → Yahoo → Google → Kite."""
    from src.data.live import live_engine

    try:
        quote = live_engine.get_quote(symbol)
        return quote.to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/depth")
def market_depth(symbol: str = Query(..., description="Ticker, e.g. RELIANCE.NS")):
    """Top-5 bid/ask market depth (NSE primary, Kite when configured)."""
    from src.data.live import live_engine

    try:
        depth = live_engine.get_depth(symbol)
        return depth.to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.get("/flow")
def market_flow(force: bool = False):
    """FII/DII net flows and index PCR from the live engine."""
    from src.data.live import live_engine

    result = {}
    pcr = live_engine.get_pcr(force=force)
    if pcr:
        result["options_pcr"] = pcr
    flows = live_engine.get_fii_dii(force=force)
    if flows:
        result["fii_dii"] = flows
        if flows.get("fii_net") is not None and flows.get("dii_net") is not None:
            result["flow_sentiment"] = get_flow_sentiment(
                flows["fii_net"], flows["dii_net"]
            )
    return result


@router.get("/pulse")
def market_pulse():
    try:
        result = {}

        fii_dii = fetch_fii_dii()
        if fii_dii is not None and not fii_dii.empty:
            latest = fii_dii.iloc[-1]
            fii_net = float(latest.get("fii_net", 0) or 0)
            dii_net = float(latest.get("dii_net", 0) or 0)
            result["fii_dii"] = {
                "fii_net": round(fii_net, 2),
                "dii_net": round(dii_net, 2),
                "flow_sentiment": get_flow_sentiment(fii_net, dii_net),
            }

        pcr = fetch_options_pcr()
        if pcr:
            result["options_pcr"] = pcr

        mtf = fetch_mtf_data("RELIANCE.NS")
        if mtf:
            combined = get_combined_signal(mtf)
            result["multi_timeframe"] = combined
            result["timeframes"] = {}
            for tf_name, tf_data in mtf.items():
                if isinstance(tf_data, dict):
                    result["timeframes"][tf_name] = tf_data

        return result
    except Exception as e:
        return {"error": str(e)}

"""FastAPI backend for StoMar React UI. v2"""

import sys
import os
import time
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    market,
    predictions,
    portfolio,
    backtest,
    scanner,
    consensus,
    sentiment,
    optimizer,
    holdings,
    risk,
    volatility,
    ranking,
    scenarios,
    regime,
    monitoring,
    pipeline,
    correlation,
    paper_trading,
    mf_tracker,
    ledger,
)

app = FastAPI(title="StoMar API", version="0.0.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_response_cache: OrderedDict[str, tuple[float, bytes]] = {}
_CACHE_TTL = 30  # seconds — short TTL for near-realtime feel

# Endpoints that are expensive and safe to cache briefly
_CACHEABLE_PREFIXES = [
    "/api/scanner", "/api/consensus", "/api/ranking",
    "/api/optimizer", "/api/correlation", "/api/risk/portfolio",
    "/api/pipeline/status", "/api/monitoring",
]


@app.middleware("http")
async def cache_middleware(request: Request, call_next):
    path = request.url.path

    if request.method == "GET" and any(path.startswith(p) for p in _CACHEABLE_PREFIXES):
        cached = _response_cache.get(path)
        if cached:
            ts, body = cached
            if time.time() - ts < _CACHE_TTL:
                return Response(content=body, media_type="application/json", headers={"X-Cache": "HIT"})

    response = await call_next(request)

    if request.method == "GET" and any(path.startswith(p) for p in _CACHEABLE_PREFIXES):
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        _response_cache[path] = (time.time(), body)
        # Evict stale entries periodically
        if len(_response_cache) > 50:
            now = time.time()
            stale = [k for k, (ts, _) in _response_cache.items() if now - ts > _CACHE_TTL * 2]
            for k in stale:
                del _response_cache[k]
        return Response(content=body, media_type="application/json", headers={"X-Cache": "MISS"})

    return response


app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(scanner.router, prefix="/api/scanner", tags=["Scanner"])
app.include_router(consensus.router, prefix="/api/consensus", tags=["Consensus"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["Sentiment"])
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"])
app.include_router(holdings.router, prefix="/api/holdings", tags=["Holdings"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk"])
app.include_router(volatility.router, prefix="/api/volatility", tags=["Volatility"])
app.include_router(ranking.router, prefix="/api/ranking", tags=["Ranking"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["Scenarios"])
app.include_router(regime.router, prefix="/api/regime", tags=["Regime"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(correlation.router, prefix="/api/correlation", tags=["Correlation"])
app.include_router(paper_trading.router, prefix="/api/paper-trading", tags=["Paper Trading"])
app.include_router(mf_tracker.router, prefix="/api/mf-tracker", tags=["MF Tracker"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["Ledger"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.0.1"}

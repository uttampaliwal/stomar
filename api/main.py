"""FastAPI backend for StoMar React UI. v2"""

import sys
import os
import hmac
import time
import uuid
from collections import OrderedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
except ImportError:
    pass

from src.core.settings import settings
from src.core.logging_config import (
    metrics, set_request_id, get_request_id, set_pipeline_context, clear_pipeline_context,
)
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

# Mutable endpoints on these prefixes require a valid API key via X-API-Key header.
_PROTECTED_PREFIXES = [
    "/api/paper-trading/",
    "/api/automation/",
    "/api/pipeline/train/",
    "/api/pipeline/run",
    "/api/ledger/",
]

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
    insights,
    automation,
    correlation,
    paper_trading,
    mf_tracker,
    ledger,
)

app = FastAPI(title="StoMar API", version="0.0.3")

if settings.env == "production":
    _cors_origins = [
        o.strip()
        for o in settings.cors_origins.split(",")
        if o.strip()
    ]
    _cors_methods = ["GET", "POST"]
    _cors_headers = ["Authorization", "Content-Type", "X-API-Key"]
else:
    _cors_origins = ["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"]
    _cors_methods = ["*"]
    _cors_headers = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=_cors_methods,
    allow_headers=_cors_headers,
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Assign a correlation ID to every request for distributed tracing."""
    request_id = request.headers.get("x-request-id", uuid.uuid4().hex[:16])
    set_request_id(request_id)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    """Collect request metrics for Prometheus export."""
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start

    path = request.url.path
    # Normalize path to avoid high-cardinality labels (skip IDs, query params)
    if path.startswith("/api/"):
        parts = path.split("/")
        normalized = "/".join(parts[:4]) if len(parts) > 4 else path
    else:
        normalized = path

    metrics.inc("http_requests_total", {
        "method": request.method,
        "path": normalized,
        "status": str(response.status_code),
    })
    metrics.observe("http_request_duration_seconds", duration, {
        "method": request.method,
        "path": normalized,
    })

    return response


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if settings.api_key and request.method in ("POST", "PUT", "PATCH", "DELETE"):
        path = request.url.path
        if any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            provided = request.headers.get("x-api-key", "")
            if not hmac.compare_digest(provided, settings.api_key):
                return Response(
                    content='{"detail":"Invalid or missing API key"}',
                    status_code=401,
                    media_type="application/json",
                )
    return await call_next(request)

_response_cache: OrderedDict[str, tuple[float, bytes]] = {}
_CACHE_TTL = settings.cache_ttl

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
        cache_key = path
        if request.url.query:
            cache_key += "?" + request.url.query
        cached = _response_cache.get(cache_key)
        if cached:
            ts, body = cached
            if time.time() - ts < _CACHE_TTL:
                metrics.inc("cache_hits_total")
                return Response(content=body, media_type="application/json", headers={"X-Cache": "HIT"})

    response = await call_next(request)

    if request.method == "GET" and any(path.startswith(p) for p in _CACHEABLE_PREFIXES):
        cache_key = path
        if request.url.query:
            cache_key += "?" + request.url.query
        body = b""
        async for chunk in response.body_iterator:
            body += chunk if isinstance(chunk, bytes) else chunk.encode()
        _response_cache[cache_key] = (time.time(), body)
        # Evict stale entries periodically
        if len(_response_cache) > settings.cache_max_entries:
            now = time.time()
            stale = [k for k, (ts, _) in _response_cache.items() if now - ts > _CACHE_TTL * 2]
            for k in stale:
                del _response_cache[k]
        metrics.inc("cache_misses_total")
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
app.include_router(insights.router, prefix="/api/insights", tags=["Insights"])
app.include_router(automation.router, prefix="/api/automation", tags=["Automation"])
app.include_router(correlation.router, prefix="/api/correlation", tags=["Correlation"])
app.include_router(paper_trading.router, prefix="/api/paper-trading", tags=["Paper Trading"])
app.include_router(mf_tracker.router, prefix="/api/mf-tracker", tags=["MF Tracker"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["Ledger"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.0.3"}


@app.get("/api/metrics")
def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )

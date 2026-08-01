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
from src.core.secure_io import atomic_append_jsonl
from src.core.logging_config import (
    metrics, set_request_id,
)
from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    market,
    market_data,
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
    wealth,
    risk_guard,
)

# Endpoints under these prefixes require a valid API key via X-API-Key header,
# regardless of HTTP method: mutating endpoints change state, and the read
# endpoints (state/positions/trades/ledger/pipeline status) expose sensitive
# account data, so both are protected fail-closed.
_PROTECTED_PREFIXES = [
    "/api/paper-trading/",
    "/api/automation/",
    "/api/pipeline/status",
    "/api/pipeline/train/",
    "/api/pipeline/run",
    "/api/ledger/",
    "/api/risk-guard/",
]

app = FastAPI(title="StoMar API", version="0.0.7")

# Fail-closed: in production, a missing STOMAR_API_KEY is a hard error.
# Without it every protected endpoint would run unauthenticated.
if settings.env == "production" and not settings.api_key:
    raise RuntimeError(
        "refusing to start in production without STOMAR_API_KEY set — "
        "all protected endpoints would be unauthenticated"
    )

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


def _is_protected_path(path: str) -> bool:
    """True when a request must carry a valid API key (fail-closed).

    Applies to every method. Reads under these prefixes leak account state
    (positions, P&L, decisions), so they are protected exactly like mutations.
    """
    return any(path.startswith(p) for p in _PROTECTED_PREFIXES)


def _auth_verdict(path: str, provided_key: str) -> tuple[int, str] | None:
    """Return (status, body) when the request must be rejected, else None.

    Fail-closed: protected endpoints are never open by configuration drift.
    """
    if not _is_protected_path(path):
        return None
    if not settings.api_key:
        return 503, '{"detail":"API key not configured; protected endpoints disabled"}'
    if not hmac.compare_digest(provided_key or "", settings.api_key):
        return 401, '{"detail":"Invalid or missing API key"}'
    return None


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    verdict = _auth_verdict(request.url.path,
                            request.headers.get("x-api-key", ""))
    if verdict is not None:
        status, body = verdict
        return Response(content=body, status_code=status, media_type="application/json")
    return await call_next(request)


# ── rate limiting (in-memory, per-IP, on protected mutating endpoints) ──

_RATE_LIMIT_MAX = 60  # requests per window
_RATE_LIMIT_WINDOW = 60.0  # seconds
_rate_buckets: dict[str, list[float]] = {}


def _rate_limited(key: str) -> bool:
    now = time.time()
    bucket = _rate_buckets.setdefault(key, [])
    bucket[:] = [ts for ts in bucket if now - ts < _RATE_LIMIT_WINDOW]
    if len(bucket) >= _RATE_LIMIT_MAX:
        return True
    bucket.append(now)
    return False


@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        path = request.url.path
        if any(path.startswith(p) for p in _PROTECTED_PREFIXES):
            client_ip = request.client.host if request.client else "unknown"
            if _rate_limited(f"{client_ip}:{request.method}"):
                return Response(
                    content='{"detail":"Rate limit exceeded"}',
                    status_code=429,
                    media_type="application/json",
                )
    return await call_next(request)


# ── audit log for every mutating request ────────────────────────────────

_AUDIT_LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "api_audit")


@app.middleware("http")
async def audit_middleware(request: Request, call_next):
    if request.method in ("POST", "PUT", "PATCH", "DELETE"):
        response = await call_next(request)
        try:
            os.makedirs(_AUDIT_LOG_DIR, exist_ok=True)
            day = time.strftime("%Y-%m-%d")
            entry = {
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "method": request.method,
                "path": request.url.path,
                "status": response.status_code,
                "client": request.client.host if request.client else "unknown",
                "request_id": request.headers.get("x-request-id", ""),
            }
            atomic_append_jsonl(
                os.path.join(_AUDIT_LOG_DIR, f"mutations-{day}.jsonl"), entry
            )
        except Exception:
            pass  # audit must never break the request path
        return response
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

    if _is_protected_path(path):
        # Never cache protected responses: this middleware runs before auth,
        # so a cached 200 (or 401) would bypass or poison the API key check.
        return await call_next(request)

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
        # Evict stale entries, then LRU if still over limit
        if len(_response_cache) > settings.cache_max_entries:
            now = time.time()
            stale = [k for k, (ts, _) in _response_cache.items() if now - ts > _CACHE_TTL * 2]
            for k in stale:
                del _response_cache[k]
            # If still over limit, evict oldest (LRU) entries
            while len(_response_cache) > settings.cache_max_entries:
                _response_cache.popitem(last=False)
        metrics.inc("cache_misses_total")
        return Response(content=body, media_type="application/json", headers={"X-Cache": "MISS"})

    return response


app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(market_data.router, prefix="/api/market-data", tags=["Market Data"])
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
app.include_router(wealth.router, prefix="/api/wealth", tags=["Wealth"])
app.include_router(risk_guard.router, prefix="/api/risk-guard", tags=["Risk Guard"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.0.7"}


@app.get("/api/metrics")
def metrics_endpoint():
    """Prometheus-compatible metrics endpoint."""
    return Response(
        content=metrics.export_prometheus(),
        media_type="text/plain; version=0.0.7; charset=utf-8",
    )

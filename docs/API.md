# API

StoMar ships a FastAPI application (`api/main.py`) exposing 73 endpoints under
`/api`, plus a React + Vite dashboard (`web/`) that consumes them.

## Running

```bash
uv run gunicorn api.main:app -w 4 -k uvicorn.workers.UvicornWorker \
      --bind 0.0.0.0:8000 --timeout 120
```

Dev: `uv run start_dev.sh` (or `.bat`) — Vite on :5173 proxies `/api` to :8000.

## Middleware stack (in order of effect)

1. **CORS** — dev: localhost:5173/3000/127.0.0.1; production: `STOMAR_CORS_ORIGINS`.
2. **Request ID** — `X-Request-ID` correlation header (echoed/assigned).
3. **Metrics** — Prometheus counters (`http_requests_total`, `http_request_duration_seconds`) at `GET /api/metrics`.
4. **Auth** — two fail-closed credential paths on paper-trading, automation, pipeline, ledger, and risk-guard routes: a valid browser session cookie (`POST /api/auth/login`, `STOMAR_AUTH_PASSWORD`, HttpOnly cookie) or the `X-API-Key` header (constant-time `hmac.compare_digest`, for CLI/scripts). Missing config → 503; wrong key/session → 401.
5. **Rate limit** — 60 requests / 60 s per IP on protected POST/PUT/PATCH/DELETE → 429; `/api/auth/login` limited to 10/min per IP.
6. **Audit** — every mutation appended to `data/api_audit/mutations-YYYY-MM-DD.jsonl`.
7. **Cache** — in-memory LRU (TTL 30 s, 50 entries) for GETs on scanner/consensus/ranking/optimizer/correlation/risk-portfolio/pipeline-status/monitoring prefixes; tagged `X-Cache: HIT|MISS`; protected paths never cached.

## Endpoints

### Market data
| Method & path | Purpose |
|---|---|
| GET `/api/market/status`, `/api/market/stocks` | Market open/closed (IST clock), tradable universe |
| GET `/api/market/history` | OHLCV history (`symbol, start, end, as_of, adjusted, interval, limit`) |
| GET `/api/market/live-quote`, `/api/market/depth`, `/api/market/flow`, `/api/market/pulse` | Live quotes, order book depth, flow, market pulse |
| GET `/api/market-data/candles`, `/quote`, `/returns`, `/sources` | Candle series, quotes, returns, enabled data sources |

### Predictions & analysis
| Method & path | Purpose |
|---|---|
| GET `/api/predictions/{ticker}` | Full prediction + confidence |
| GET `/api/predictions/ensemble/{ticker}` | Ensemble breakdown |
| GET `/api/predictions/{ticker}/feature-importance`, `/explain` | Top features / SHAP explanation |
| GET `/api/scanner/`, `/api/consensus/`, `/api/ranking/` | Universe scans and consensus ranking |
| GET `/api/sentiment/{ticker}` | News/sentiment score |
| GET `/api/optimizer/`, `/optimizer/advanced` | Portfolio optimization |
| GET `/api/volatility/{ticker}`, `/api/scenarios/{ticker}`, `/api/regime/{ticker}`, `/api/risk/{ticker}` | Vol, scenarios, regime, per-ticker risk |
| GET `/api/correlation/`, `/api/holdings/stats` | Correlation matrix, holdings |
| GET `/api/backtest/{ticker}` | Historical backtest |
| GET `/api/insights/recommend/{ticker}`, `/enrichment/{ticker}` | Recommendation + enrichment |
| GET `/api/mf-tracker/portfolio` | Mutual-fund style tracking |

### Paper trading (protected)
| Method & path | Purpose |
|---|---|
| GET `/api/paper-trading/performance`, `/state`, `/positions`, `/trades`, `/stocks` | Full paper account view |
| POST `/api/paper-trading/order` | Place order (ticker regex `^[A-Z0-9]{1,20}\.NS$`, qty 1–1,000,000, gated) |
| POST `/api/paper-trading/reset` | Reset account (capital 1–100,000,000) |
| POST `/api/paper-trading/close-position` | Close a position |

Paper-trading API **refuses to run when trading mode is `live`**.

### Pipeline (protected)
| Method & path | Purpose |
|---|---|
| GET `/api/pipeline/status` | Current pipeline status |
| POST `/api/pipeline/train/{ticker}` + GET `/status` | Train one ticker, poll progress |
| POST `/api/pipeline/run` + GET `/run/status` | Full pipeline run, poll progress (≤50 tickers) |
| POST `/api/automation/run`, GET `/api/automation/decisions` | Auto-run + decisions |

### Ledger (protected)
| Method & path | Purpose |
|---|---|
| GET `/api/ledger/decisions`, `/trades`, `/summary` | Stored decisions, trades, summary |
| GET `/api/ledger/performance` | Equity curve + returns |
| GET `/api/ledger/benchmark` | Paper vs Nifty 50 buy-and-hold (alpha, IR, drawdowns) |
| GET `/api/ledger/calibration`, `/agreement-stats` | Calibration (ECE/MCE), model agreement |

### Monitoring
| Method & path | Purpose |
|---|---|
| GET `/api/monitoring/` | Alerts + trained models |
| GET `/api/monitoring/failures` | Pipeline failures (24 h) |
| GET `/api/monitoring/metrics`, `/retrain-triggers`, `/checkpoint` | Metrics, drift retrain triggers, checkpoint |
| GET `/api/monitoring/readiness` | Live-readiness gate (60 paper days, accuracy ≥51%, drawdown ≤15%) |
| GET `/api/monitoring/daily-health` | Health sweep JSON |

### Risk guard (protected)
| Method & path | Purpose |
|---|---|
| GET `/api/risk-guard/status`, `/scalar` | Freeze state, volatility scalar |
| POST `/api/risk-guard/resume` | Resume after freeze (`override_token`, 403 on mismatch) |
| POST `/api/risk-guard/equity`, `/pnl` | Manual equity/P&L updates |

### Other
- GET `/api/health` — liveness probe.
- GET `/api/metrics` — Prometheus text format.
- POST `/api/wealth/monte-carlo`, `/api/wealth/wealth-advisor` — wealth simulations (Gemini-backed advisor, falls back offline).

## Authentication

Protected prefixes: `/api/paper-trading/`, `/api/automation/`, `/api/pipeline/`,
`/api/ledger/`, `/api/risk-guard/` (27 endpoints).

**Browser sessions (default for the UI):** `POST /api/auth/login` with
`{"password": "<STOMAR_AUTH_PASSWORD>"}` sets an HttpOnly session cookie
(`stomar_session`); `GET /api/auth/me` reports `{"authenticated": true|false}`;
`POST /api/auth/logout` destroys the session. Sessions expire after
`STOMAR_SESSION_TTL_HOURS` (default 12).

**API key (CLI/scripts/curl):** send the key in every request:

```
X-API-Key: <STOMAR_API_KEY>
```

In `production` the app refuses to boot without at least one of
`STOMAR_AUTH_PASSWORD` / `STOMAR_API_KEY`. The React app uses sessions only —
no key is bundled into the frontend.

## Errors

- Routers mostly return HTTP 200 with `{"error": "..."}`; `HTTPException` for 400 (bad params), 422 (validation), 403 (bad override token), 429 (rate limit), 503 (unconfigured auth / live source down).
- Every response carries `X-Request-ID`.

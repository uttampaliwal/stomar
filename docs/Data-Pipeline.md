# Data Pipeline

From raw market feeds to training-ready, point-in-time-correct features.

## Sources & fetching (`src/data/`)

- **Primary: yfinance** (`fetch_free_historical_data`, adjusted OHLCV).
- **Fallback: NSE archive** — monthly bhavcopy zips from `archives.nseindia.com`.
- Universe: `NSE_STOCKS` = **20 large-caps** (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK, HINDUNILVR, ITC, SBIN, BHARTIARTL, KOTAKBANK, BAJFINANCE, LT, WIPRO, AXISBANK, TITAN, MARUTI, SUNPHARMA, ASIANPAINT, NTPC, ONGC). Extended universes (NIFTY 50 / NEXT 50 / MIDCAP 100) available in `src/data/universes.py`.
- Default fetch: **2 years, daily bars**; adjusted prices (split/dividend aware).
- Caching: 10-min in-memory TTL; parquet files `data/{TICKER}.parquet` refreshed when stale >2 days or below expected row counts.
- Auto-repair (`fetch_validated_stock_data`): corporate-action re-fetch → extended-period gap re-fetch → forward-fill **non-price columns only** (never OHLC). Repairs logged to `data/monitoring/data_fixes.json`.
- Resilience: circuit breakers per source — `yf_breaker` (5 failures / 120 s), `nse_breaker` (3 failures / 90 s); exponential backoff + jitter; token-bucket rate limiting (yfinance 5/s, NSE 2/s).

## Point-in-time store (`src/data/store.py`)

DuckDB at `data/market_data.db`:

- Tables: `ohlcv_daily`, `minute_bars`, `corporate_actions` (splits/bonus/dividends with ratios), `options_chain`, `quote_snapshots`, `depth_snapshots`, `universe_snapshot`.
- **Raw (unadjusted) bars are stored**; adjustment computed at query time with `get_history(as_of=...)` — only bars `date <= as_of` and corporate actions `ex_date <= as_of` are used, eliminating look-ahead bias.
- Corporate actions synced from yfinance and the NSE API.

## Trading calendar (`src/core/calendar.py`)

- `is_trading_day()` = weekday **and** not an NSE holiday (2025: 16, 2026: 15 holidays).
- `trading_days_since(start, end)` powers gap detection and backfill.
- Unknown years fall back to Mon–Fri (holiday misses cost one extra fetch, never halt trading).

## Backfill (`src/core/backfill.py`)

- `HistoricalBackfill.run(tickers, lookback_days=252)` reconstructs past decisions to seed meta-controller training.
- Requires ≥60 rows + 60-day warmup per ticker; regime recomputed day-by-day on data up to day *i* only.
- Unavailable signals (sentiment, FII/DII, PCR) get neutral defaults; MTF approximated via SMA20/50 heuristic.
- **Idempotent** — skips dates already in the ledger; outcomes tagged `source="backfill"` (in-sample by construction, excluded from trust metrics).
- Decision rule: weighted score (ensemble ±0.4, sentiment ±0.2, regime ±0.1, MTF ±0.15, VaR −0.1); BUY/SELL at |score| > 0.15; size `min(0.05, |score|*0.15)`.

## Features (`src/data/features.py`, `src/signals/feature_pipeline.py`)

- SOTA factor pipeline: **8 factor groups** (trend, momentum, volatility, volume, microstructure, returns, external, calendar) → **~60 features**.
- Indicators: SMA/EMA, RSI(14), MACD, Bollinger, ATR(14), OBV, Stoch, Williams %R, CCI, MFI, ADX, VWAP, supertrend(10, 3.0), Ichimoku, CMF(20), vol z-score(20), momentum 21/63/126/252d, relative strength vs NIFTY 50.
- Targets: 1-day forward return + direction; triple-barrier labels ±2% / 5-bar max holding.
- **Point-in-time discipline**: sentiment/PCR assigned only to the latest row; FII/DII shifted +1 day (published after T close); banned look-ahead feature `ichimoku_chikou` neutralized at inference.

## Feature store / versioning (`src/data/feature_store.py`)

- 12-char SHA-256 of sorted column list + transformations → `data/feature_versions/{hash}.json`.
- `verify_feature_compatibility()` compares the model's schema hash to the current one, so stale models are flagged rather than silently used.
- Current live registry: hash `13a6f5fb80d8`, 60 features (2026-08-04).

## Validation & retention

- `src/data/data_validation.py` — gap auto-fill (P2.1).
- `src/data/data_retention.py` — 7-step cleanup:
  - Cache files > 7 days (protected files: stomar.db, paper_state.json, mf_state.json, paper_session.json, options_pcr.json, fii_dii.parquet).
  - Ledger entries > 90 days archived to `data/archive/*.parquet`, then deleted + VACUUM.
  - Monitoring files > 30 days; feature versions keep 10; pipeline logs > 14 days; stale-ticker files; orphaned checkpoints.
  - All overridable via `STOMAR_*` env vars.

## Live market data

- `src/data/live.py` — quote polling (default 3 s), NSE universe, optional Kite WebSocket when both Kite credentials are set.
- Market status via IST clock: open 09:15–15:30 IST, weekdays.

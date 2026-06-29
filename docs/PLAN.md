# Stage 3 — Live Data + Scheduling: Execution Plan

## Status: Stage 2 Complete (Signal NOT significant → pivot to volatility/ranking done)

Stage 3 answers the question: **Can we get reliable data, retrain automatically, and catch model degradation before it costs money?**

This stage is about infrastructure — not improving the signal. The signal was proven noise in Stage 2 (p > 0.05). We built volatility/ranking/regime tools as the pivot. Stage 3 makes the system production-grade so that when we DO find a real edge (or use the volatility tools), the data and pipelines are trustworthy.

---

## Stage 3 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | Primary data feed with SLA (validation, fallback) | ❌ NOT DONE | CRITICAL |
| 2 | Data validation (gaps, splits, dividends checked) | ❌ NOT DONE | CRITICAL |
| 3 | Feature store with versioning | ❌ NOT DONE | HIGH |
| 4 | Nightly retraining pipeline running | ❌ NOT DONE | HIGH |
| 5 | Monitoring dashboard with drift alerts | ❌ NOT DONE | HIGH |
| 6 | Model registry with promotion logic | ❌ NOT DONE | HIGH |

---

## Pre-Stage 3: Current State Assessment

| Component | Current State | What Needs to Change |
|-----------|--------------|---------------------|
| Data source | yfinance only, no validation | Add validation, fallback, gap detection |
| Cache | Parquet files, stale after 2 days | TTL-based invalidation, hash integrity |
| Features | 41 features, computed on-the-fly | Version-pinned, deterministic hashing |
| Training | Manual "Train" button in UI | Nightly cron with validation gates |
| Monitoring | None | Track OOS metrics, drift, alerting |
| Model registry | Files on disk (`models/`) | Versioned, with promotion logic |
| Logging | Python `logging` module | Structured JSON logs for ML pipeline |

---

## Execution Plan: 6 Tasks

### Task 1: Data Validation Layer
**Impact:** Critical — garbage in, garbage out
**Effort:** 4-5 hours
**Files:** New `src/data_validation.py`, update `src/data_fetcher.py`

**Why this matters:**
yfinance for NSE is unreliable — it silently drops dates, doesn't handle splits/dividends properly, and can return stale data. Without validation, we train on incomplete data and don't know it.

**What to implement:**

#### 1a. Gap Detection
```python
def detect_gaps(df: pd.DataFrame, expected_freq: str = "B") -> list:
    """
    Find missing trading days in price data.
    
    Args:
        df: DataFrame with DatetimeIndex
        expected_freq: Expected frequency ('B' = business days)
    
    Returns:
        List of (start, end, n_days) tuples for each gap
    """
    # Reindex to expected frequency
    full_range = pd.date_range(df.index.min(), df.index.max(), freq=expected_freq)
    missing = full_range.difference(df.index)
    
    # Group consecutive missing dates into gaps
    gaps = []
    if len(missing) > 0:
        gap_starts = [missing[0]]
        for i in range(1, len(missing)):
            if (missing[i] - missing[i-1]).days > 3:  # New gap
                gaps.append((gap_starts[-1], missing[i-1], i - gap_starts.index(gap_starts[-1])))
        gaps.append((gap_starts[-1], missing[-1], len(missing) - gap_starts.index(gap_starts[-1])))
    
    return gaps
```

#### 1b. Corporate Action Detection (Splits/Dividends)
```python
def detect_corporate_actions(df: pd.DataFrame, threshold: float = 0.15) -> list:
    """
    Detect potential stock splits or dividends by looking for
    unnatural price jumps (>15% overnight without volume spike).
    
    Returns:
        List of (date, type, old_price, new_price) tuples
    """
    returns = df["close"].pct_change()
    volume_change = df["volume"].pct_change()
    
    suspicious = []
    for date in df.index[1:]:
        ret = abs(returns.loc[date])
        vol_ratio = volume_change.loc[date] if date in volume_change.index else 0
        
        if ret > threshold:
            # Large price move with low volume = likely corporate action
            if vol_ratio < 2.0:  # Volume didn't spike
                suspicious.append({
                    "date": date,
                    "type": "split_or_dividend" if returns.loc[date] > 0 else "reverse_split",
                    "return": returns.loc[date],
                    "volume_change": vol_ratio,
                })
    
    return suspicious
```

#### 1c. Stale Data Detection
```python
def detect_stale_data(df: pd.DataFrame, max_age_days: int = 2) -> dict:
    """
    Check if the data is stale (last date is too old).
    
    Returns:
        Dict with is_stale, last_date, age_days
    """
    last_date = df.index[-1]
    now = pd.Timestamp.now()
    if last_date.tz is not None:
        now = now.tz_localize(last_date.tz)
    
    age_days = (now - last_date).days
    return {
        "is_stale": age_days > max_age_days,
        "last_date": str(last_date.date()),
        "age_days": age_days,
        "max_age_days": max_age_days,
    }
```

#### 1d. Price Sanity Checks
```python
def validate_prices(df: pd.DataFrame) -> list:
    """
    Check for impossible prices (negative, zero, >100% daily moves).
    
    Returns:
        List of validation errors
    """
    errors = []
    
    # Negative or zero prices
    for col in ["open", "high", "low", "close"]:
        bad = df[df[col] <= 0]
        if len(bad) > 0:
            errors.append(f"{col} has {len(bad)} non-positive values")
    
    # High < Low
    bad_hl = df[df["high"] < df["low"]]
    if len(bad_hl) > 0:
        errors.append(f"high < low on {len(bad_hl)} days")
    
    # Close outside high-low range
    bad_close = df[(df["close"] > df["high"]) | (df["close"] < df["low"])]
    if len(bad_close) > 0:
        errors.append(f"close outside high-low range on {len(bad_close)} days")
    
    # Extreme daily moves (>30%)
    returns = df["close"].pct_change()
    extreme = df[abs(returns) > 0.30]
    if len(extreme) > 0:
        errors.append(f"Extreme daily moves (>30%) on {len(extreme)} days")
    
    return errors
```

#### 1e. Full Validation Pipeline
```python
def validate_data(df: pd.DataFrame, ticker: str) -> dict:
    """
    Run all validations on price data.
    
    Returns:
        Dict with passed, errors, warnings, gaps, corporate_actions
    """
    errors = []
    warnings = []
    
    # Price sanity
    price_errors = validate_prices(df)
    errors.extend(price_errors)
    
    # Gap detection
    gaps = detect_gaps(df)
    if gaps:
        total_missing = sum(g[2] for g in gaps)
        warnings.append(f"{len(gaps)} gaps found, {total_missing} missing trading days")
    
    # Stale data
    stale = detect_stale_data(df)
    if stale["is_stale"]:
        warnings.append(f"Data is {stale['age_days']} days old (last: {stale['last_date']})")
    
    # Corporate actions
    corp_actions = detect_corporate_actions(df)
    if corp_actions:
        warnings.append(f"{len(corp_actions)} potential corporate actions detected")
    
    return {
        "ticker": ticker,
        "passed": len(errors) == 0,
        "errors": errors,
        "warnings": warnings,
        "gaps": gaps,
        "corporate_actions": corp_actions,
        "stale_info": stale,
        "data_points": len(df),
        "date_range": f"{df.index[0].date()} to {df.index[-1].date()}",
    }
```

#### 1f. Integrate with data_fetcher.py
- Call `validate_data()` after every fetch
- Log validation results
- Surface warnings in the UI (Trader tab)
- Return validation metadata alongside the DataFrame

**Tests:** `tests/test_data_validation.py`
- `test_detect_gaps_empty` — no gaps in clean data
- `test_detect_gaps_found` — correctly identifies gaps
- `test_detect_corporate_actions` — detects splits
- `test_validate_prices_clean` — clean data passes
- `test_validate_prices_negative` — catches negative prices
- `test_validate_prices_extreme_move` — catches >30% moves
- `test_detect_stale_data` — old data flagged
- `test_validate_data_full` — end-to-end validation
- `test_validate_data_with_gaps` — gap + price check combined

---

### Task 2: Multi-Source Data with Fallback
**Impact:** High — single point of failure with yfinance only
**Effort:** 3-4 hours
**Files:** New `src/data_sources.py`, update `src/data_fetcher.py`

**Why this matters:**
yfinance is rate-limited, can be down, and sometimes returns wrong data for NSE stocks. A fallback source prevents silent failures.

**Free data sources for Indian market:**

| Source | Type | Reliability | Rate Limit |
|--------|------|-------------|------------|
| yfinance (current) | Primary | Medium | 2000 req/hr |
| NSE website (archives) | Fallback | Medium (breaks often) | No official limit |
| Upstox API (free tier) | Alternative | High | 5000 req/day |

**Implementation:**

```python
# src/data_sources.py

class DataSource:
    """Base class for data sources."""
    
    def fetch(self, ticker: str, period: str, interval: str) -> pd.DataFrame:
        raise NotImplementedError
    
    def validate(self, df: pd.DataFrame) -> bool:
        raise NotImplementedError


class YFinanceSource(DataSource):
    """Primary source: yfinance."""
    
    def fetch(self, ticker, period="2y", interval="1d"):
        import yfinance as yf
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        # ... standardize columns
        return df
    
    def validate(self, df):
        return len(df) > 50 and not df.empty


class NSEArchiveSource(DataSource):
    """Fallback: NSE India archive data."""
    
    def fetch(self, ticker, period="2y", interval="1d"):
        # Download from NSE archives (if available)
        # Convert to same format as yfinance
        ...
    
    def validate(self, df):
        return len(df) > 50 and not df.empty


def fetch_with_fallback(ticker, period="2y", interval="1d"):
    """
    Try primary source, fall back to secondary if it fails.
    
    Returns:
        Dict with df, source, validation, timestamp
    """
    sources = [YFinanceSource(), NSEArchiveSource()]
    
    for source in sources:
        try:
            df = source.fetch(ticker, period, interval)
            if source.validate(df):
                validation = validate_data(df, ticker)
                return {
                    "df": df,
                    "source": source.__class__.__name__,
                    "validation": validation,
                    "timestamp": pd.Timestamp.now(),
                }
        except Exception as e:
            logger.warning(f"{source.__class__.__name__} failed for {ticker}: {e}")
            continue
    
    raise ValueError(f"All data sources failed for {ticker}")
```

**Tests:** `tests/test_data_sources.py`
- `test_yfinance_source_fetch` — basic fetch works
- `test_yfinance_source_validate` — validation logic
- `test_fallback_on_failure` — falls back to secondary
- `test_all_sources_fail` — raises ValueError
- `test_fetch_returns_metadata` — returns source + validation

---

### Task 3: Feature Store with Versioning
**Impact:** High — prevents training/serving skew
**Effort:** 4-5 hours
**Files:** New `src/feature_store.py`, update `src/features.py`, update `src/model.py`

**Why this matters:**
If we train a model on v1 features but serve predictions using v3 features (different column order, different transforms), the model silently breaks. A feature store pins feature definitions to model versions.

**Implementation:**

```python
# src/feature_store.py

import hashlib
import json
import os
from datetime import datetime

FEATURE_VERSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "feature_versions"
)


def compute_feature_hash(feature_cols: list, transformations: dict = None) -> str:
    """
    Deterministic hash of feature definition.
    
    Args:
        feature_cols: Sorted list of feature column names
        transformations: Dict of column -> transformation applied
    
    Returns:
        12-char hex hash
    """
    content = json.dumps({
        "columns": sorted(feature_cols),
        "transformations": transformations or {},
    }, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def register_feature_version(
    version_hash: str,
    feature_cols: list,
    transformations: dict = None,
    description: str = "",
) -> dict:
    """
    Register a feature version with metadata.
    
    Returns:
        Version record dict
    """
    os.makedirs(FEATURE_VERSIONS_DIR, exist_ok=True)
    
    record = {
        "hash": version_hash,
        "columns": sorted(feature_cols),
        "n_features": len(feature_cols),
        "transformations": transformations or {},
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    
    # Save version record
    path = os.path.join(FEATURE_VERSIONS_DIR, f"{version_hash}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    
    # Also save as "latest" if it's the first or explicitly marked
    latest_path = os.path.join(FEATURE_VERSIONS_DIR, "latest.json")
    if not os.path.exists(latest_path):
        with open(latest_path, "w") as f:
            json.dump(record, f, indent=2)
    
    return record


def load_feature_version(version_hash: str) -> dict:
    """Load a feature version record."""
    path = os.path.join(FEATURE_VERSIONS_DIR, f"{version_hash}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature version {version_hash} not found")
    with open(path) as f:
        return json.load(f)


def verify_feature_compatibility(
    model_version_hash: str, current_feature_hash: str
) -> dict:
    """
    Check if current features match what the model was trained on.
    
    Returns:
        Dict with compatible, model_hash, current_hash, differences
    """
    try:
        model_record = load_feature_version(model_version_hash)
    except FileNotFoundError:
        return {
            "compatible": False,
            "error": f"Model feature version {model_version_hash} not found",
        }
    
    if model_version_hash == current_feature_hash:
        return {"compatible": True, "model_hash": model_version_hash, "current_hash": current_feature_hash}
    
    model_cols = set(model_record["columns"])
    current_cols = set(json.loads(
        open(os.path.join(FEATURE_VERSIONS_DIR, f"{current_feature_hash}.json")).read()
    )["columns"])
    
    return {
        "compatible": False,
        "model_hash": model_version_hash,
        "current_hash": current_feature_hash,
        "missing_in_current": sorted(model_cols - current_cols),
        "extra_in_current": sorted(current_cols - model_cols),
    }
```

#### Integration with model.py
- When saving a model, save the feature version hash alongside it
- When loading a model, verify feature compatibility
- Log a warning if features have drifted

#### Integration with features.py
- `add_technical_indicators` returns a `feature_hash` alongside the DataFrame
- Hash is computed from the sorted list of columns produced

**Tests:** `tests/test_feature_store.py`
- `test_compute_feature_hash_deterministic` — same input = same hash
- `test_compute_feature_hash_different` — different input = different hash
- `test_register_and_load` — save/load roundtrip
- `test_verify_compatibility_match` — same hash = compatible
- `test_verify_compatibility_mismatch` — different hash = incompatible
- `test_verify_compatibility_missing` — missing columns detected
- `test_verify_compatibility_extra` — extra columns detected

---

### Task 4: Automated Retraining Pipeline
**Impact:** Critical — models go stale without retraining
**Effort:** 5-6 hours
**Files:** New `src/pipeline.py`, update `src/trainer.py`

**Why this matters:**
Currently, models are trained once via a UI button and never updated. Market regimes change, features drift, and stale models lose whatever edge they had. An automated pipeline with validation gates prevents deploying bad models.

**Architecture:**

```
┌─────────────────────────────────────────────────┐
│                 PIPELINE STAGE                   │
│                                                  │
│  1. FETCH  →  2. VALIDATE  →  3. FEATURES  →  4. TRAIN  │
│                                                  │
│  5. EVALUATE  →  6. DECISION  →  7. PROMOTE/REJECT  │
└─────────────────────────────────────────────────┘
```

**Implementation:**

```python
# src/pipeline.py

import logging
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for the retraining pipeline."""
    tickers: list = field(default_factory=lambda: [
        "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"
    ])
    lookback_period: str = "3y"
    min_oos_accuracy: float = 0.50    # Minimum OOS accuracy to promote
    min_oos_sharpe: float = -1.0      # Maximum acceptable negative Sharpe
    max_drawdown_threshold: float = 0.30  # 30% max drawdown to reject
    retrain_cooldown_days: int = 7    # Don't retrain same ticker within 7 days


@dataclass
class PipelineResult:
    """Result of a single pipeline run."""
    ticker: str
    stage: str
    status: str  # "success", "failed", "rejected"
    message: str
    metrics: dict = field(default_factory=dict)
    model_path: Optional[str] = None
    feature_hash: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RetrainingPipeline:
    """
    Automated retraining pipeline with validation gates.
    
    Stages:
    1. Fetch latest data
    2. Validate data quality
    3. Compute features
    4. Train models
    5. Evaluate on OOS window
    6. Decision: promote or reject
    7. Log everything
    """
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.results = []
    
    def run(self, ticker: str) -> PipelineResult:
        """Run the full pipeline for a single ticker."""
        logger.info(f"Starting retraining pipeline for {ticker}")
        
        # Stage 1: Fetch
        result = self._fetch_data(ticker)
        if result.status == "failed":
            return result
        
        # Stage 2: Validate
        result = self._validate_data(ticker, result)
        if result.status == "failed":
            return result
        
        # Stage 3: Features
        result = self._compute_features(ticker, result)
        if result.status == "failed":
            return result
        
        # Stage 4: Train
        result = self._train_model(ticker, result)
        if result.status == "failed":
            return result
        
        # Stage 5: Evaluate
        result = self._evaluate_model(ticker, result)
        if result.status in ("failed", "rejected"):
            return result
        
        # Stage 6: Promote
        result = self._promote_model(ticker, result)
        
        logger.info(f"Pipeline complete for {ticker}: {result.status}")
        self.results.append(result)
        return result
    
    def _fetch_data(self, ticker: str) -> PipelineResult:
        """Stage 1: Fetch latest price data."""
        try:
            from src.data_fetcher import fetch_stock_data
            df = fetch_stock_data(ticker, period=self.config.lookback_period, force_refresh=True)
            return PipelineResult(
                ticker=ticker, stage="fetch", status="success",
                message=f"Fetched {len(df)} rows",
                metrics={"rows": len(df), "date_range": f"{df.index[0]} to {df.index[-1]}"},
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="fetch", status="failed",
                message=f"Fetch failed: {e}",
            )
    
    def _validate_data(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 2: Validate data quality."""
        try:
            from src.data_validation import validate_data
            df = fetch_stock_data(ticker, period=self.config.lookback_period)
            validation = validate_data(df, ticker)
            
            if not validation["passed"]:
                return PipelineResult(
                    ticker=ticker, stage="validate", status="failed",
                    message=f"Validation failed: {validation['errors']}",
                    metrics=validation,
                )
            
            status = "success"
            message = f"Validation passed ({validation['data_points']} points)"
            if validation["warnings"]:
                message += f" with {len(validation['warnings'])} warnings"
            
            return PipelineResult(
                ticker=ticker, stage="validate", status=status,
                message=message, metrics=validation,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="validate", status="failed",
                message=f"Validation error: {e}",
            )
    
    def _compute_features(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 3: Compute features with versioning."""
        try:
            from src.features import add_technical_indicators
            from src.feature_store import compute_feature_hash, register_feature_version
            
            df = fetch_stock_data(ticker, period=self.config.lookback_period)
            df = add_technical_indicators(df, ticker)
            
            feature_cols = [c for c in df.columns if c not in (
                "open", "high", "low", "close", "volume", "target", "target_direction"
            )]
            feature_hash = compute_feature_hash(feature_cols)
            
            register_feature_version(
                feature_hash, feature_cols,
                description=f"Auto-registered for {ticker} pipeline run"
            )
            
            return PipelineResult(
                ticker=ticker, stage="features", status="success",
                message=f"Computed {len(feature_cols)} features (hash: {feature_hash})",
                metrics={"n_features": len(feature_cols), "feature_hash": feature_hash},
                feature_hash=feature_hash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="features", status="failed",
                message=f"Feature computation failed: {e}",
            )
    
    def _train_model(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 4: Train all 5 models."""
        try:
            from src.trainer import train_for_ticker
            
            result = train_for_ticker(ticker, epochs=40, save=True)
            
            return PipelineResult(
                ticker=ticker, stage="train", status="success",
                message=f"Training complete for {ticker}",
                metrics=result.get("metrics", {}),
                feature_hash=prev.feature_hash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="train", status="failed",
                message=f"Training failed: {e}",
            )
    
    def _evaluate_model(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 5: Evaluate on OOS window."""
        try:
            from src.backtester import run_walk_forward_backtest
            
            results = run_walk_forward_backtest(ticker)
            metrics = results.get("metrics", {})
            
            oos_accuracy = metrics.get("ensemble_accuracy", 0)
            sharpe = metrics.get("simulated_sharpe", -999)
            max_dd = metrics.get("max_drawdown", 1.0)
            
            # Validation gates
            reasons = []
            if oos_accuracy < self.config.min_oos_accuracy:
                reasons.append(f"OOS accuracy {oos_accuracy:.1%} < {self.config.min_oos_accuracy:.1%}")
            if sharpe < self.config.min_oos_sharpe:
                reasons.append(f"Sharpe {sharpe:.2f} < {self.config.min_oos_sharpe}")
            if max_dd > self.config.max_drawdown_threshold:
                reasons.append(f"Max DD {max_dd:.1%} > {self.config.max_drawdown_threshold:.1%}")
            
            if reasons:
                return PipelineResult(
                    ticker=ticker, stage="evaluate", status="rejected",
                    message=f"Model rejected: {'; '.join(reasons)}",
                    metrics=metrics,
                )
            
            return PipelineResult(
                ticker=ticker, stage="evaluate", status="success",
                message=f"Model passed: accuracy={oos_accuracy:.1%}, sharpe={sharpe:.2f}",
                metrics=metrics,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="evaluate", status="failed",
                message=f"Evaluation failed: {e}",
            )
    
    def _promote_model(self, ticker: str, prev: PipelineResult) -> PipelineResult:
        """Stage 6: Promote model to production."""
        try:
            from src.model import promote_model
            
            promote_model(ticker, prev.feature_hash)
            
            return PipelineResult(
                ticker=ticker, stage="promote", status="success",
                message=f"Model promoted for {ticker}",
                metrics=prev.metrics,
                model_path=f"models/{ticker.replace('.', '_')}_production.pt",
                feature_hash=prev.feature_hash,
            )
        except Exception as e:
            return PipelineResult(
                ticker=ticker, stage="promote", status="failed",
                message=f"Promotion failed: {e}",
                metrics=prev.metrics,
            )
    
    def run_all(self) -> list:
        """Run pipeline for all configured tickers."""
        results = []
        for ticker in self.config.tickers:
            result = self.run(ticker)
            results.append(result)
            logger.info(f"{ticker}: {result.status} — {result.message}")
        return results
    
    def get_summary(self) -> dict:
        """Get summary of all pipeline runs."""
        return {
            "total": len(self.results),
            "success": sum(1 for r in self.results if r.status == "success"),
            "failed": sum(1 for r in self.results if r.status == "failed"),
            "rejected": sum(1 for r in self.results if r.status == "rejected"),
            "results": [
                {"ticker": r.ticker, "stage": r.stage, "status": r.status, "message": r.message}
                for r in self.results
            ],
        }
```

#### CLI Entry Point
```python
# run_pipeline.py (new file)
if __name__ == "__main__":
    import sys
    from src.pipeline import RetrainingPipeline, PipelineConfig
    
    tickers = sys.argv[1:] if len(sys.argv) > 1 else None
    config = PipelineConfig(tickers=tickers or ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"])
    
    pipeline = RetrainingPipeline(config)
    results = pipeline.run_all()
    
    summary = pipeline.get_summary()
    print(f"\nPipeline Summary: {summary['success']} success, {summary['failed']} failed, {summary['rejected']} rejected")
    
    for r in results:
        emoji = "✅" if r.status == "success" else ("❌" if r.status == "failed" else "⚠️")
        print(f"  {emoji} {r.ticker}: [{r.stage}] {r.message}")
```

**Tests:** `tests/test_pipeline.py`
- `test_pipeline_stages_exist` — all 6 stages callable
- `test_fetch_stage` — fetch returns success
- `test_validate_stage_clean_data` — clean data passes
- `test_validate_stage_bad_data` — bad data fails
- `test_evaluate_rejects_low_accuracy` — accuracy gate works
- `test_evaluate_rejects_low_sharpe` — sharpe gate works
- `test_evaluate_rejects_high_drawdown` — drawdown gate works
- `test_evaluate_accepts_good_model` — good model passes
- `test_run_all` — runs for multiple tickers
- `test_get_summary` — summary counts correct

---

### Task 5: Monitoring & Drift Detection
**Impact:** High — catches model degradation early
**Effort:** 4-5 hours
**Files:** New `src/monitoring.py`

**Why this matters:**
Models degrade silently. A model that was 53% accurate last month might be 48% this month. Without monitoring, you don't know until you've lost money.

**What to monitor:**

| Metric | What It Catches | Threshold |
|--------|----------------|-----------|
| OOS accuracy drift | Model degradation | < 48% accuracy |
| Feature distribution shift | Data regime change | KS test p < 0.01 |
| Prediction distribution shift | Model bias shift | >10% shift in mean prediction |
| Return distribution change | Strategy breakdown | Sharpe drops below 0 |
| Data freshness | Stale data pipeline | >3 days old |

**Implementation:**

```python
# src/monitoring.py

import logging
import json
import os
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

MONITORING_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "data", "monitoring"
)


@dataclass
class DriftAlert:
    """A single drift detection alert."""
    metric: str
    severity: str  # "info", "warning", "critical"
    message: str
    current_value: float
    threshold: float
    p_value: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class MonitoringReport:
    """Complete monitoring report for a ticker."""
    ticker: str
    alerts: list = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    @property
    def has_critical(self) -> bool:
        return any(a.severity == "critical" for a in self.alerts)
    
    @property
    def has_warnings(self) -> bool:
        return any(a.severity == "warning" for a in self.alerts)
    
    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "timestamp": self.timestamp,
            "has_critical": self.has_critical,
            "has_warnings": self.has_warnings,
            "n_alerts": len(self.alerts),
            "alerts": [
                {"metric": a.metric, "severity": a.severity, "message": a.message}
                for a in self.alerts
            ],
            "metrics": self.metrics,
        }


class ModelMonitor:
    """
    Monitor model health and data drift.
    
    Detects:
    1. Performance degradation (OOS accuracy dropping)
    2. Feature drift (distribution shift in input features)
    3. Prediction drift (model output distribution shifting)
    4. Data freshness (stale data in pipeline)
    """
    
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days
        os.makedirs(MONITORING_DIR, exist_ok=True)
    
    def check_performance_drift(
        self, ticker: str, current_accuracy: float, baseline_accuracy: float
    ) -> DriftAlert:
        """Check if model accuracy has degraded."""
        drift = baseline_accuracy - current_accuracy
        drift_pct = drift / baseline_accuracy if baseline_accuracy > 0 else 0
        
        if drift_pct > 0.10:  # >10% relative degradation
            severity = "critical"
        elif drift_pct > 0.05:  # >5% relative degradation
            severity = "warning"
        else:
            severity = "info"
        
        alert = DriftAlert(
            metric="oos_accuracy",
            severity=severity,
            message=f"OOS accuracy: {current_accuracy:.1%} (baseline: {baseline_accuracy:.1%}, drift: {drift_pct:.1%})",
            current_value=current_accuracy,
            threshold=baseline_accuracy * 0.90,
        )
        
        self._log_alert(ticker, alert)
        return alert
    
    def check_feature_drift(
        self, ticker: str, baseline_features: np.ndarray, current_features: np.ndarray
    ) -> DriftAlert:
        """
        Check for distribution shift in features using KS test.
        
        Args:
            baseline_features: Feature matrix from training period
            current_features: Feature matrix from recent period
        """
        # KS test on each feature
        n_features = min(baseline_features.shape[1], current_features.shape[1])
        p_values = []
        
        for i in range(n_features):
            _, p_value = stats.ks_2samp(baseline_features[:, i], current_features[:, i])
            p_values.append(p_value)
        
        mean_p = np.mean(p_values)
        n_drifted = sum(1 for p in p_values if p < 0.01)
        
        if n_drifted > n_features * 0.3:  # >30% of features drifted
            severity = "critical"
        elif n_drifted > n_features * 0.15:  # >15% drifted
            severity = "warning"
        else:
            severity = "info"
        
        alert = DriftAlert(
            metric="feature_drift",
            severity=severity,
            message=f"Feature drift: {n_drifted}/{n_features} features shifted (mean KS p={mean_p:.3f})",
            current_value=n_drifted / n_features,
            threshold=0.15,
            p_value=mean_p,
        )
        
        self._log_alert(ticker, alert)
        return alert
    
    def check_prediction_drift(
        self, ticker: str, baseline_predictions: np.ndarray, current_predictions: np.ndarray
    ) -> DriftAlert:
        """Check for distribution shift in model predictions."""
        baseline_mean = np.mean(baseline_predictions)
        current_mean = np.mean(current_predictions)
        
        shift = abs(current_mean - baseline_mean)
        
        if shift > 0.15:  # >15% shift in mean prediction
            severity = "critical"
        elif shift > 0.10:  # >10% shift
            severity = "warning"
        else:
            severity = "info"
        
        alert = DriftAlert(
            metric="prediction_drift",
            severity=severity,
            message=f"Prediction drift: mean shifted from {baseline_mean:.3f} to {current_mean:.3f} (Δ={shift:.3f})",
            current_value=current_mean,
            threshold=baseline_mean + 0.10,
        )
        
        self._log_alert(ticker, alert)
        return alert
    
    def check_data_freshness(self, ticker: str, last_data_date: datetime) -> DriftAlert:
        """Check if data is fresh enough."""
        now = datetime.now()
        age_days = (now - last_data_date).days
        
        if age_days > 7:
            severity = "critical"
        elif age_days > 3:
            severity = "warning"
        else:
            severity = "info"
        
        alert = DriftAlert(
            metric="data_freshness",
            severity=severity,
            message=f"Data is {age_days} days old (last: {last_data_date.date()})",
            current_value=age_days,
            threshold=3,
        )
        
        self._log_alert(ticker, alert)
        return alert
    
    def generate_report(self, ticker: str) -> MonitoringReport:
        """Generate a full monitoring report for a ticker."""
        report = MonitoringReport(ticker=ticker)
        
        # Load baseline metrics if available
        baseline_path = os.path.join(MONITORING_DIR, f"{ticker.replace('.', '_')}_baseline.json")
        if os.path.exists(baseline_path):
            with open(baseline_path) as f:
                baseline = json.load(f)
            report.metrics["baseline"] = baseline
        
        # Store current report
        report_path = os.path.join(MONITORING_DIR, f"{ticker.replace('.', '_')}_latest.json")
        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2)
        
        return report
    
    def _log_alert(self, ticker: str, alert: DriftAlert):
        """Log an alert and store it."""
        level = {"info": logging.INFO, "warning": logging.WARNING, "critical": logging.CRITICAL}
        logger.log(level.get(alert.severity, logging.INFO), f"[{ticker}] {alert.message}")
        
        # Append to alert history
        history_path = os.path.join(MONITORING_DIR, f"{ticker.replace('.', '_')}_alerts.json")
        alerts = []
        if os.path.exists(history_path):
            with open(history_path) as f:
                alerts = json.load(f)
        
        alerts.append({
            "metric": alert.metric,
            "severity": alert.severity,
            "message": alert.message,
            "current_value": alert.current_value,
            "threshold": alert.threshold,
            "p_value": alert.p_value,
            "timestamp": alert.timestamp,
        })
        
        # Keep last 100 alerts
        alerts = alerts[-100:]
        
        with open(history_path, "w") as f:
            json.dump(alerts, f, indent=2)
```

**Tests:** `tests/test_monitoring.py`
- `test_check_performance_drift_none` — no drift = info
- `test_check_performance_drift_warning` — 5-10% drift = warning
- `test_check_performance_drift_critical` — >10% drift = critical
- `test_check_feature_drift_clean` — no drift = info
- `test_check_feature_drift_detected` — KS test catches shift
- `test_check_prediction_drift_clean` — no shift = info
- `test_check_prediction_drift_detected` — large shift caught
- `test_check_data_freshness_fresh` — recent data = info
- `test_check_data_freshness_stale` — old data = critical
- `test_generate_report` — report contains expected fields
- `test_alert_history_persisted` — alerts saved to disk

---

### Task 6: Model Registry with Promotion Logic
**Impact:** High — prevents accidentally deploying bad models
**Effort:** 3-4 hours
**Files:** New `src/model_registry.py`, update `src/model.py`

**Why this matters:**
Right now, training a model overwrites the previous version. If the new model is worse, we lose the old one. A registry keeps version history and requires explicit promotion.

**Implementation:**

```python
# src/model_registry.py

import os
import json
import shutil
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

REGISTRY_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "models", "registry"
)


@dataclass
class ModelVersion:
    """A single model version record."""
    ticker: str
    version: int
    model_path: str
    feature_hash: str
    metrics: dict = field(default_factory=dict)
    status: str = "staging"  # "staging", "production", "archived"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    promoted_at: Optional[str] = None
    notes: str = ""


class ModelRegistry:
    """
    Version-controlled model registry.
    
    Workflow:
    1. Train → save as "staging" version
    2. Evaluate → if passes gates, promote to "production"
    3. Old "production" → archived
    4. Load always loads "production" version
    """
    
    def __init__(self):
        os.makedirs(REGISTRY_DIR, exist_ok=True)
    
    def _registry_path(self, ticker: str) -> str:
        return os.path.join(REGISTRY_DIR, f"{ticker.replace('.', '_')}_registry.json")
    
    def _load_registry(self, ticker: str) -> list:
        path = self._registry_path(ticker)
        if os.path.exists(path):
            with open(path) as f:
                return json.load(f)
        return []
    
    def _save_registry(self, ticker: str, versions: list):
        path = self._registry_path(ticker)
        with open(path, "w") as f:
            json.dump(versions, f, indent=2)
    
    def register(
        self, ticker: str, model_path: str, feature_hash: str,
        metrics: dict = None, notes: str = ""
    ) -> ModelVersion:
        """Register a new model version as staging."""
        versions = self._load_registry(ticker)
        
        # Determine next version number
        max_version = max((v["version"] for v in versions), default=0)
        new_version = max_version + 1
        
        version_record = ModelVersion(
            ticker=ticker,
            version=new_version,
            model_path=model_path,
            feature_hash=feature_hash,
            metrics=metrics or {},
            status="staging",
            notes=notes,
        )
        
        versions.append(version_record.__dict__)
        self._save_registry(ticker, versions)
        
        return version_record
    
    def promote(self, ticker: str, version: int) -> ModelVersion:
        """Promote a staging model to production."""
        versions = self._load_registry(ticker)
        
        target = None
        for v in versions:
            if v["version"] == version:
                target = v
                break
        
        if target is None:
            raise ValueError(f"Version {version} not found for {ticker}")
        
        if target["status"] != "staging":
            raise ValueError(f"Version {version} is {target['status']}, not staging")
        
        # Archive current production
        for v in versions:
            if v["status"] == "production":
                v["status"] = "archived"
        
        # Promote new
        target["status"] = "production"
        target["promoted_at"] = datetime.now().isoformat()
        
        self._save_registry(ticker, versions)
        
        # Copy model file to production path
        production_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "models", f"{ticker.replace('.', '_')}_production.pt"
        )
        shutil.copy2(target["model_path"], production_path)
        
        return ModelVersion(**target)
    
    def get_production(self, ticker: str) -> Optional[ModelVersion]:
        """Get the current production model version."""
        versions = self._load_registry(ticker)
        for v in versions:
            if v["status"] == "production":
                return ModelVersion(**v)
        return None
    
    def get_version(self, ticker: str, version: int) -> Optional[ModelVersion]:
        """Get a specific version."""
        versions = self._load_registry(ticker)
        for v in versions:
            if v["version"] == version:
                return ModelVersion(**v)
        return None
    
    def list_versions(self, ticker: str) -> list:
        """List all versions for a ticker."""
        versions = self._load_registry(ticker)
        return [ModelVersion(**v) for v in versions]
    
    def rollback(self, ticker: str, to_version: int) -> ModelVersion:
        """Rollback to a previous production version."""
        versions = self._load_registry(ticker)
        
        target = None
        for v in versions:
            if v["version"] == to_version and v["status"] == "archived":
                target = v
                break
        
        if target is None:
            raise ValueError(f"Archived version {to_version} not found for {ticker}")
        
        return self.promote(ticker, to_version)
```

**Tests:** `tests/test_model_registry.py`
- `test_register_new_version` — version 1 created
- `test_register_multiple` — versions increment
- `test_promote_model` — status changes to production
- `test_promote_archives_old` — old production archived
- `test_promote_non_staging_fails` — can't promote non-staging
- `test_get_production` — returns current production
- `test_list_versions` — returns all versions
- `test_rollback` — rolls back to archived version
- `test_rollback_nonexistent_fails` — error on missing version

---

## Execution Order

```
1. Task 1: Data Validation (4-5 hrs)     ← CRITICAL, foundation for everything
2. Task 2: Multi-Source Fallback (3-4 hrs) ← depends on Task 1 validation
3. Task 3: Feature Store (4-5 hrs)         ← independent, but benefits from Task 1
4. Task 4: Retraining Pipeline (5-6 hrs)   ← depends on Tasks 1, 2, 3
5. Task 5: Monitoring (4-5 hrs)            ← depends on Task 4 (needs baseline)
6. Task 6: Model Registry (3-4 hrs)        ← depends on Task 4 (needs versioning)
```

**Parallelizable:**
- Tasks 1, 2, 3 can be developed in parallel (different concerns)
- Tasks 5, 6 can be developed in parallel (both depend on Task 4)

**Total estimated time:** 23-29 hours

---

## Exit Criteria Checklist

After all 6 tasks:

- [ ] Data validation catches gaps, splits, stale data, price errors
- [ ] Fallback data source works when primary fails
- [ ] Feature store pins feature definitions to model versions
- [ ] Retraining pipeline runs with validation gates (accuracy, Sharpe, drawdown)
- [ ] Monitoring detects performance drift, feature drift, prediction drift
- [ ] Model registry versions models with staging → production → archived lifecycle
- [ ] All 6 tasks have tests passing
- [ ] `run_pipeline.py` works as CLI entry point
- [ ] Monitoring dashboard shows alerts in the UI (optional: add to a new tab)

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/data_validation.py` | Gap detection, corporate action detection, price sanity, stale data |
| `src/data_sources.py` | Multi-source with fallback (yfinance + NSE archive) |
| `src/feature_store.py` | Feature versioning, hash computation, compatibility check |
| `src/pipeline.py` | 6-stage retraining pipeline with validation gates |
| `src/monitoring.py` | Drift detection (performance, features, predictions, freshness) |
| `src/model_registry.py` | Version registry with staging/production/archived lifecycle |
| `run_pipeline.py` | CLI entry point for automated retraining |
| `tests/test_data_validation.py` | 9 tests for data validation |
| `tests/test_data_sources.py` | 5 tests for multi-source fallback |
| `tests/test_feature_store.py` | 7 tests for feature versioning |
| `tests/test_pipeline.py` | 10 tests for retraining pipeline |
| `tests/test_monitoring.py` | 11 tests for drift detection |
| `tests/test_model_registry.py` | 9 tests for model registry |

## Files to Modify

| File | Change |
|------|--------|
| `src/data_fetcher.py` | Integrate `validate_data()` after fetch, add fallback source |
| `src/features.py` | Return `feature_hash` alongside DataFrame |
| `src/model.py` | Save feature_hash with model, verify compatibility on load |
| `src/trainer.py` | Return metrics dict for pipeline integration |
| `app.py` | Add "Pipeline" and "Monitoring" tabs (optional) |
| `requirements.txt` | Add `scipy>=1.11` (already present) |

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130, scipy installed)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run linter:** `C:\Users\uttam\venv\Scripts\ruff.exe check src/ tests/ --select F401,E,F,W --ignore E501,F841`
- **Current test count:** 221 tests passing
- **Existing modules:** `data_fetcher.py`, `features.py`, `model.py`, `trainer.py`, `logging_config.py`
- **Pipeline will be triggered by:** `python run_pipeline.py` or cron schedule
- **User constraint:** Everything must be free (no Kite Connect at ₹500/month)
- **Data strategy:** yfinance primary + NSE archive fallback (both free)
- **Key insight:** This stage is about INFRASTRUCTURE, not signal improvement. The signal was proven noise in Stage 2.

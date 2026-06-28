# Stage 1 — Hardening: Execution Plan

## Status Assessment

**We are NOT ready for Stage 1 yet.** The cleanup phase (imports, constants, tests, CI, Docker) is complete, but the actual Stage 1 implementation from IMPROVEMENTS.md is still pending.

### What's Done (Cleanup Phase)
- ✅ `.gitignore` — comprehensive (models/*.pt, data/*.pkl, data/*.json)
- ✅ `requirements.txt` — all deps including scipy, requests
- ✅ `src/constants.py` — shared constants (BROKERAGE_RATE, RISK_FREE_RATE, etc.)
- ✅ Unused imports cleaned — 15 removed across 9 files
- ✅ Lambda → def fixes — 3 in model.py
- ✅ 49 tests passing — backtester, features, model, portfolio, risk
- ✅ CI pipeline — `.github/workflows/ci.yml`
- ✅ Dockerfile — Python 3.12 slim
- ✅ `pyproject.toml` — project config, pytest, ruff

### What's NOT Done (Stage 1 Actual Work)
- ❌ **Point-in-Time Data Discipline** — features.py has look-ahead bias
- ❌ **Realistic NSE Cost Model** — constants exist but portfolio.py/backtester.py use simple costs
- ❌ **Structured Logging** — 20 print() calls, zero logging module usage
- ❌ **Expanded Test Coverage** — missing point-in-time tests, sentiment tests, data_fetcher tests

---

## Stage 1 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | All leakage points in `features.py` and `flow.py` identified and fixed | ❌ NOT DONE | CRITICAL |
| 2 | Test suite passes with >80% coverage on critical modules | ⚠️ 49 tests, no coverage measurement | HIGH |
| 3 | GitHub Actions CI runs on every push | ✅ DONE | - |
| 4 | Realistic NSE cost model implemented | ❌ NOT DONE | HIGH |
| 5 | Structured logging in place | ❌ NOT DONE | MEDIUM |
| 6 | Docker builds and runs successfully | ✅ DONE | - |

---

## Execution Plan: 6 Tasks

### Task 1: Point-in-Time Data Fixes (CRITICAL)
**Impact:** The single most common reason backtests lie
**Effort:** 2-3 hours
**Files:** `src/features.py`

**The Problem:**
All alternative data features are joined to the SAME day's price row, creating look-ahead bias:

| Feature | Current (BAD) | Fixed (GOOD) |
|---------|---------------|--------------|
| `sentiment_score` | Today's news → today's price | Yesterday's news → today's price |
| `fii_net` | Today's FII data → today's price | Yesterday's FII data → today's price |
| `dii_net` | Today's DII data → today's price | Yesterday's DII data → today's price |
| `flow_signal` | Computed from today's flow | Computed from yesterday's flow |
| `pcr` | Today's options PCR → today's price | Yesterday's PCR → today's price |
| `mtf_signal` | Uses today's close in calculation | Use only data available before today |
| `mtf_confidence` | Uses today's close in calculation | Use only data available before today |

**Concrete Fix:**
```python
# In add_flow_features() — after reading fii_dii.parquet:
# SHIFT all flow features by 1 day to prevent look-ahead
df["fii_net"] = fii_data["fii_net"].shift(1)  # T+1 lag
df["dii_net"] = fii_data["dii_net"].shift(1)
df["flow_signal"] = ...  # computed from shifted values

# In add_sentiment_features() — sentiment from today's news
# should only be available AFTER market close
df["sentiment_score"] = score  # Already OK if computed from prev day's cache
# BUT: verify the cache is from yesterday, not today

# In add_pcr_features() — PCR from market hours
df["pcr"] = pcr  # Need to verify cache timestamp
```

**Verification:**
- Write `test_point_in_time()` that asserts no feature uses same-day data
- Write `test_feature_lag()` that verifies `.shift(1)` is applied

**Files to modify:** `src/features.py` (lines 8-84)
**Files to create:** `tests/test_point_in_time.py`

---

### Task 2: Realistic NSE Cost Model (HIGH)
**Impact:** Backtests lie without realistic costs
**Effort:** 1-2 hours
**Files:** `src/portfolio.py`, `src/backtester.py`, `src/constants.py`

**Current State:**
- `src/constants.py` has `NSE_TRANSACTION_COSTS` dict with all rates
- `portfolio.py` uses only `brokerage=0.0003` (ignores STT, stamp duty, etc.)
- `backtester.py` uses only `brokerage=0.0003, slippage=0.001`

**What to Implement:**

1. **Add `calculate_nse_costs()` to `src/constants.py`:**
```python
def calculate_nse_costs(price: float, quantity: int, side: str) -> dict:
    """
    Calculate full NSE transaction costs for Indian equity delivery trades.
    
    Args:
        price: Execution price per share
        quantity: Number of shares
        side: "buy" or "sell"
    
    Returns:
        Dict with cost breakdown and total
    """
    trade_value = price * quantity
    
    brokerage = trade_value * BROKERAGE_RATE  # 0.03%
    exchange_charge = trade_value * EXCHANGE_CHARGE_RATE
    sebi_fees = trade_value * SEBI_FEES_RATE
    gst = (brokerage + exchange_charge) * GST_RATE  # 18% on brokerage + exchange
    
    if side == "sell":
        stt = trade_value * STT_SELL_RATE  # 0.1% on sell
        stamp_duty = 0
    else:
        stt = 0
        stamp_duty = trade_value * STAMP_DUTY_BUY_RATE  # 0.015% on buy
    
    total = brokerage + stt + exchange_charge + sebi_fees + stamp_duty + gst
    
    return {
        "brokerage": brokerage,
        "stt": stt,
        "exchange_charge": exchange_charge,
        "sebi_fees": sebi_fees,
        "stamp_duty": stamp_duty,
        "gst": gst,
        "total": total,
        "effective_rate": total / trade_value if trade_value > 0 else 0,
    }
```

2. **Update `portfolio.py` `buy()` and `sell()` to use full NSE costs:**
```python
# Current (SIMPLIFIED):
broker_fee = cost * brokerage

# New (REALISTIC):
from src.constants import calculate_nse_costs
costs = calculate_nse_costs(price, quantity, "buy")
total_cost = cost + costs["total"]
```

3. **Update `backtester.py` to use realistic costs:**
```python
# Current:
portfolio.buy(ticker, exec_price, qty, date, brokerage)

# New:
portfolio.buy(ticker, exec_price, qty, date)  # costs calculated internally
```

**Verification:**
- Write `test_nse_costs_buy()` — verify buy-side costs
- Write `test_nse_costs_sell()` — verify sell-side costs (includes STT)
- Write `test_nse_round_trip()` — verify total round-trip cost is ~0.3-0.4%
- Write `test_portfolio_buy_with_full_costs()`
- Write `test_portfolio_sell_with_full_costs()`

**Files to modify:** `src/constants.py`, `src/portfolio.py`, `src/backtester.py`
**Files to create:** `tests/test_nse_costs.py`

---

### Task 3: Structured Logging (MEDIUM)
**Impact:** Debugging Streamlit state is painful without logs
**Effort:** 1-2 hours
**Files:** All `src/*.py`, `app.py`

**Current State:** 20 `print()` calls across trainer.py and backtester.py, zero `logging` module usage.

**What to Implement:**

1. **Create `src/logging_config.py`:**
```python
import logging
import json
import sys

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "module": record.module,
            "message": record.getMessage(),
        })

def setup_logging(level="INFO"):
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logging.basicConfig(level=level, handlers=[handler])
    return logging.getLogger("stomar")
```

2. **Replace all `print()` calls with `logger.info()` / `logger.debug()`:**
```python
# Before:
print(f"Training models for {ticker}")

# After:
logger = logging.getLogger("stomar.trainer")
logger.info("training_started ticker=%s", ticker)
```

3. **Log key events:**
- Training start/end with duration
- Prediction confidence
- Backtest results
- Cost calculations
- Errors with full context

**Files to modify:** `src/trainer.py` (18 prints), `src/backtester.py` (1 print), all other src/*.py
**Files to create:** `src/logging_config.py`

---

### Task 4: Expanded Test Coverage (HIGH)
**Impact:** Verify all Stage 1 fixes work correctly
**Effort:** 2-3 hours
**Files:** `tests/`

**Current:** 49 tests covering backtester, features, model, portfolio, risk.
**Missing tests per IMPROVEMENTS.md:**

| Test | Module | What It Verifies |
|------|--------|------------------|
| `test_point_in_time` | features | No future data in features |
| `test_feature_lag` | features | Alternative data shifted by 1 day |
| `test_sentiment_range` | features | Score in [-1, 1] |
| `test_ensemble_direction` | ensemble | Output is 0 or 1 |
| `test_confidence_range` | ensemble | Output in [0, 100] |
| `test_fetch_returns_dataframe` | data_fetcher | yfinance → DataFrame |
| `test_cache_hit` | data_fetcher | Second fetch uses cache |
| `test_market_status` | data_fetcher | Returns "Open" or "Closed" |
| `test_nse_costs_buy` | constants | Buy-side cost calculation |
| `test_nse_costs_sell` | constants | Sell-side cost calculation |
| `test_nse_round_trip` | constants | Total round-trip ~0.3-0.4% |
| `test_walk_forward_no_leakage` | backtester | Verify no future data in training |
| `test_train_test_no_overlap` | backtester | Train and test sets disjoint |

**Coverage Target:** >80% on critical modules (backtester, features, risk, portfolio)

**Files to create:** `tests/test_point_in_time.py`, `tests/test_nse_costs.py`, `tests/test_ensemble.py`, `tests/test_data_fetcher.py`
**Files to modify:** `tests/test_backtester.py`, `tests/test_features.py`

---

### Task 5: Verify CI Pipeline (LOW)
**Impact:** Automated quality gate
**Effort:** 15 minutes
**Files:** `.github/workflows/ci.yml`

**Already done.** Just verify it works by checking the YAML is valid and the workflow would run.

**Verification:**
- Check YAML syntax
- Confirm pytest + ruff commands are correct
- Verify Python version matches (3.12 not 3.14 since CI runner may not have 3.14)

**Files to modify:** `.github/workflows/ci.yml` (minor: Python version)

---

### Task 6: Verify Docker (LOW)
**Impact:** Reproducibility
**Effort:** 15 minutes
**Files:** `Dockerfile`

**Already done.** Just verify it builds.

**Verification:**
- Check Dockerfile syntax
- Confirm base image exists
- Verify healthcheck command
- Test build locally (if Docker available)

---

## Execution Order

```
1. Task 1: Point-in-Time Fixes          (2-3 hrs)  ← CRITICAL, do first
2. Task 2: NSE Cost Model               (1-2 hrs)  ← HIGH, do second
3. Task 4: Expanded Tests               (2-3 hrs)  ← HIGH, after Tasks 1-2
4. Task 3: Structured Logging           (1-2 hrs)  ← MEDIUM, can be parallel
5. Task 5: Verify CI                    (15 min)   ← LOW, quick check
6. Task 6: Verify Docker                (15 min)   ← LOW, quick check
```

**Total estimated time:** 8-12 hours

---

## Exit Criteria Checklist

After all 6 tasks are complete:

- [ ] All leakage points in `features.py` and `flow.py` identified and fixed
- [ ] Test suite passes with >80% coverage on critical modules
- [ ] GitHub Actions CI runs on every push
- [ ] Realistic NSE cost model implemented
- [ ] Structured logging in place
- [ ] Docker builds and runs successfully

**Only when ALL 6 are checked can we say Stage 1 is complete and move to Stage 2 (Alpha & Backtest Rigor).**

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130, streamlit 1.58.0)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run linter:** `ruff check src/ --select F401,E,F,W --ignore E501,F841`
- **Streamlit command:** `python -m streamlit run app.py` (not `streamlit run`)
- **`load_models` returns 7 values:** `(lstm, gru, transformer, xgb, scaler, features, lgb_model)`

# Stage 2 — Alpha & Backtest Rigor: Execution Plan

## Status: Stage 1 Complete (122 tests, 99% risk coverage, 91% portfolio, 81% features)

Stage 2 answers the question: **Is the 45-50% signal real or noise?**

---

## Stage 2 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | CPCV implemented and run on all 3 trained stocks | ❌ NOT DONE | CRITICAL |
| 2 | Deflated Sharpe computed (accounting for how many variations were tried) | ❌ NOT DONE | CRITICAL |
| 3 | Permutation test shows real performance is outside null distribution (p < 0.05) | ❌ NOT DONE | HIGH |
| 4 | All metrics reported with confidence intervals | ❌ NOT DONE | HIGH |
| 5 | Performance compared to buy-and-hold and momentum benchmarks | ❌ NOT DONE | HIGH |
| 6 | Alpha research pipeline defined for each signal type | ❌ NOT DONE | MEDIUM |
| 7 | **Decision: Is the signal real?** If yes → Stage 3. If no → pivot. | ❌ NOT DONE | CRITICAL |

---

## Execution Plan: 6 Tasks

### Task 1: Combinatorial Purged Cross-Validation (CPCV)
**Impact:** Critical — catches false discoveries in backtests
**Effort:** 3-4 hours
**Files:** New `src/significance.py`

**Why CPCV over standard walk-forward:**
- Walk-forward gives a single point estimate of performance
- CPCV gives the DISTRIBUTION of outcomes (probability of ruin)
- Purging removes overlapping labels between train/test
- Embargoing adds a gap to prevent leakage through autocorrelation

**Implementation:**
```python
# src/significance.py

def purged_kfold_cv(df, feature_cols, target_col, n_splits=6, 
                    embargo_pct=0.01, purge_window=5):
    """
    Walk-forward CV with purging and embargoing.
    
    1. Split data into n_splits chronological folds
    2. For each fold: train on all prior folds, test on current
    3. Purge: remove last `purge_window` rows from training 
       (they overlap with test labels)
    4. Embargo: skip `embargo_pct` of data between train/test
    """
    ...

def combinatorial_cv(df, feature_cols, target_col, n_test_groups=3):
    """
    Generate all C(n_test_groups, test_size) combinations.
    For each combination, compute performance metric.
    Returns distribution of outcomes.
    """
    ...
```

**Key detail from López de Prado:**
- Purging: If label at time t uses data from t-5 to t, and test starts at t+1,
  we must remove training samples t-4 through t from training set
- Embargo: Add gap of `embargo_pct * n_samples` between train and test

**Test:** `tests/test_significance.py::TestPurgedCV`

---

### Task 2: Deflated Sharpe Ratio
**Impact:** Critical — adjusts for multiple testing bias
**Effort:** 2-3 hours (after CPCV)
**Files:** `src/significance.py`

**The problem:**
If you tested 100 strategies and found one with Sharpe 2.0, the deflated Sharpe might be 0.8 — noise. The more strategies you try, the more you need to deflate.

**Implementation:**
```python
def deflated_sharpe_ratio(sharpe_observed, n_trials, sharpe_max, T, skew, kurtosis):
    """
    Adjusts observed Sharpe for multiple testing bias.
    
    From: López de Prado, "The Deflated Sharpe Ratio" (2014)
    
    Args:
        sharpe_observed: Your best strategy's Sharpe
        n_trials: How many strategy variations you tried
        sharpe_max: Maximum possible Sharpe (theoretical bound)
        T: Number of independent observations (trades)
        skew: Return distribution skewness
        kurtosis: Return distribution excess kurtosis
    
    Returns:
        Deflated Sharpe ratio and p-value
    """
    # E[max(SR)] under null (no skill)
    euler_mascheroni = 0.5772156649
    e_max_sr = sharpe_max * (
        (1 - euler_mascheroni) * norm.ppf(1 - 1/n_trials) + 
        euler_mascheroni * norm.ppf(1 - 1/(n_trials * np.e))
    )
    
    # Adjusted standard error of Sharpe
    sr_std = np.sqrt(
        (1 + 0.5 * sharpe_observed**2 - skew * sharpe_observed + 
         (kurtosis - 3) / 4 * sharpe_observed**2) / T
    )
    
    # Deflated Sharpe
    deflated = (sharpe_observed - e_max_sr) / sr_std
    p_value = 1 - norm.cdf(deflated)
    
    return deflated, p_value
```

**What to count as "n_trials":**
- 5 models × 3 hyperparameter sets × 4 feature combinations = 60 trials
- Or: count all experiments run in this project = conservative estimate

**Test:** `tests/test_significance.py::TestDeflatedSharpe`

---

### Task 3: Permutation Test
**Impact:** High — most honest test of signal reality
**Effort:** 1-2 hours
**Files:** `src/significance.py`

**Implementation:**
```python
def permutation_test_accuracy(predictions, actuals, n_permutations=1000):
    """
    Test whether observed accuracy is significantly above chance.
    
    1. Compute real accuracy
    2. Shuffle actuals n_permutations times
    3. Compute accuracy on each shuffle → null distribution
    4. p-value = fraction of null accuracies >= real accuracy
    """
    real_accuracy = np.mean(predictions == actuals)
    
    null_accuracies = []
    for _ in range(n_permutations):
        shuffled = np.random.permutation(actuals)
        null_acc = np.mean(predictions == shuffled)
        null_accuracies.append(null_acc)
    
    null_accuracies = np.array(null_accuracies)
    p_value = np.mean(null_accuracies >= real_accuracy)
    
    return {
        "real_accuracy": real_accuracy,
        "null_mean": null_accuracies.mean(),
        "null_std": null_accuracies.std(),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }
```

**Test:** `tests/test_significance.py::TestPermutationTest`

---

### Task 4: Benchmarks & Comparison
**Impact:** High — context for performance
**Effort:** 2-3 hours
**Files:** `src/backtester.py`, new `src/benchmarks.py`

**Implement 4 benchmarks:**
```python
# src/benchmarks.py

def benchmark_buy_and_hold(df, initial_capital=100000):
    """Passive buy-and-hold strategy."""
    ...

def benchmark_20day_momentum(df, initial_capital=100000):
    """Go long if price > 20-day SMA, else flat."""
    ...

def benchmark_always_long(initial_capital=100000):
    """Always in the market."""
    ...

def benchmark_random(df, n_trades, initial_capital=100000):
    """Random entries/exits with same trade frequency."""
    ...
```

**Comparison metrics:**
- Annualized return
- Sharpe ratio
- Sortino ratio
- Max drawdown
- Win rate

**Test:** `tests/test_benchmarks.py`

---

### Task 5: Confidence Intervals on All Metrics
**Impact:** Medium — shows uncertainty
**Effort:** 1-2 hours
**Files:** `src/significance.py`

**Implementation:**
```python
def bootstrap_confidence_interval(data, metric_fn, n_bootstrap=1000, 
                                  confidence=0.95):
    """
    Bootstrap confidence interval for any metric.
    
    Returns: (point_estimate, ci_lower, ci_upper)
    """
    point = metric_fn(data)
    boot_stats = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_stats.append(metric_fn(sample))
    
    alpha = 1 - confidence
    ci_lower = np.percentile(boot_stats, 100 * alpha / 2)
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
    
    return point, ci_lower, ci_upper
```

**Apply to:**
- Accuracy: 45-50% → 95% CI
- Sharpe: X.XX → 95% CI
- Annual return: XX% → 95% CI

**Test:** `tests/test_significance.py::TestBootstrapCI`

---

### Task 6: Alpha Research Pipeline
**Impact:** Medium — structured experimentation process
**Effort:** 2-3 hours
**Files:** New `src/alpha_research.py`

**Implementation:**
```python
# src/alpha_research.py

SIGNAL_HYPOTHESES = {
    "sentiment": {
        "hypothesis": "Positive sentiment predicts next-day UP",
        "feature": "sentiment_score",
        "direction": "positive",
    },
    "fii_flow": {
        "hypothesis": "FII net buying predicts next-day UP",
        "feature": "fii_net",
        "direction": "positive",
    },
    "pcr": {
        "hypothesis": "High PCR predicts next-day UP (contrarian)",
        "feature": "pcr",
        "direction": "negative",
    },
    "rsi": {
        "hypothesis": "Low RSI predicts next-day UP (mean reversion)",
        "feature": "rsi",
        "direction": "negative",
    },
    "macd": {
        "hypothesis": "MACD crossover predicts next-day UP",
        "feature": "macd",
        "direction": "positive",
    },
}

def test_signal_hypothesis(df, signal_name, hypothesis):
    """Test a single signal's predictive power standalone."""
    ...

def test_signal_decay(df, signal_name, windows=[5, 10, 20, 60]):
    """How quickly does signal predictive power decay?"""
    ...

def rank_signals(df, signals=None):
    """Rank all signals by standalone predictive power."""
    ...
```

**Output:** Table showing each signal's standalone accuracy, Sharpe, and decay rate.

**Test:** `tests/test_alpha_research.py`

---

## Execution Order

```
1. Task 1: CPCV (3-4 hrs)          ← CRITICAL, foundation for Tasks 2-5
2. Task 2: Deflated Sharpe (2-3 hrs) ← depends on Task 1
3. Task 3: Permutation Test (1-2 hrs) ← can be parallel with Task 2
4. Task 4: Benchmarks (2-3 hrs)     ← independent
5. Task 5: Confidence Intervals (1-2 hrs) ← depends on Task 1
6. Task 6: Alpha Research (2-3 hrs) ← independent
```

**Total estimated time:** 12-17 hours

---

## Exit Criteria Checklist

After all 6 tasks:

- [ ] CPCV implemented and run on all 3 trained stocks
- [ ] Deflated Sharpe computed (accounting for how many variations were tried)
- [ ] Permutation test shows real performance is outside null distribution (p < 0.05)
- [ ] All metrics reported with confidence intervals
- [ ] Performance compared to buy-and-hold and momentum benchmarks
- [ ] Alpha research pipeline defined for each signal type
- [ ] **Decision: Is the signal real?** If yes → Stage 3. If no → pivot.

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run coverage:** `python -m pytest tests/ --cov=src --cov-report=term-missing`
- **Run linter:** `ruff check src/ tests/ --select F401,E,F,W --ignore E501,F841`
- **122 tests currently passing**

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/significance.py` | CPCV, Deflated Sharpe, Permutation Test, Bootstrap CI |
| `src/benchmarks.py` | Buy-and-hold, momentum, always-long, random strategies |
| `src/alpha_research.py` | Hypothesis testing, signal decay, signal ranking |
| `tests/test_significance.py` | Tests for CPCV, deflated Sharpe, permutation, CI |
| `tests/test_benchmarks.py` | Tests for benchmark strategies |
| `tests/test_alpha_research.py` | Tests for alpha research pipeline |

## Files to Modify

| File | Change |
|------|--------|
| `src/backtester.py` | Add `compare_to_benchmarks()` function |
| `app.py` | Add "Significance" tab with CPCV/deflated Sharpe/permutation results |

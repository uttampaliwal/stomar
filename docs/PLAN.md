# Stage 5 — Open Platform: Execution Plan

## Status: Stage 4 Complete (Event-driven engine, paper trading, risk controls, execution quality — 424 tests)

Stage 5 answers the question: **Can we make the ensemble smarter, cover the full NSE universe, integrate mutual funds, and prepare for open-source release?**

This stage upgrades the modeling (learned ensemble weights, regime routing), expands coverage (all 20 stocks), adds MF tracking, improves sentiment, and prepares the project for others to use.

---

## Stage 5 Exit Criteria (from IMPROVEMENTS.md)

| # | Criterion | Status | Priority |
|---|-----------|--------|----------|
| 1 | Stacked meta-learner (learned ensemble weights) | ✅ DONE | CRITICAL |
| 2 | Regime-conditional routing (different weights per regime) | ✅ DONE | HIGH |
| 3 | Volatility forecasting or ranking prototype built | ✅ DONE (fundamental ranking + volatility module) | MEDIUM |
| 4 | MF NAV integration + XIRR computation | ✅ DONE (mf_tracker.py with NAV, XIRR, factors) | HIGH |
| 5 | Sentiment upgraded with multi-source NLP | ✅ DONE (5 sources: Yahoo, Google, MoneyControl, ET, Screener) | MEDIUM |
| 6 | All 20 stocks trained (batch_train) | ✅ DONE (batch_train + --train-all flag) | MEDIUM |
| 7 | Open-source license + contribution guidelines | ✅ DONE (Apache 2.0 + CONTRIBUTING.md) | LOW |

---

## Pre-Stage 5: Current State Assessment

| Component | Current State | What Needs to Change |
|-----------|--------------|---------------------|
| Ensemble | Equal-weighted 5-model (20% each) | Learned meta-learner (LogisticRegression on model outputs) |
| Regime routing | Rule-based detection, separate strategy | Route through ensemble — different model weights per regime |
| Sentiment | FinBERT on Google News headlines, 30-min cache | Multi-source (news + social + transcripts), FinGPT option |
| Holdings/MF | Zerodha CSV import, category allocation | Full MF tracker: NAV history, factor exposures, AMFI data |
| Ranking | Technical factors only (momentum/vol/volume/technical) | Add fundamental factors (P/E, P/B, ROE, dividend yield) |
| Universe | 3 of 20 stocks trained | All 20 stocks, batch training |
| Volatility | 5 estimators + mean-reversion forecast | Already solid — enhance with GARCH if time permits |
| Licensing | None | MIT license + CONTRIBUTING.md |

---

## Design Decisions

### Why LogisticRegression for meta-learner (not neural net)?

| Factor | LogisticRegression | Neural Meta-Learner |
|--------|-------------------|---------------------|
| Interpretability | Coefficients show model contribution | Black box |
| Overfitting risk | Low (linear model) | High (small dataset: ~250 trading days) |
| Training speed | Instant | Minutes |
| Maintenance | Simple | Requires GPU, tuning |
| Performance | Competitive for 5 inputs | Marginal improvement not worth complexity |

**Decision:** Use LogisticRegression with L2 regularization. The meta-learner has only 5 inputs (one per base model) and ~250 training samples per walk-forward window. A linear model is the right complexity level.

### Why not build FinGPT from scratch?

FinGPT is a 7B parameter model. Fine-tuning requires:
- GPU with 16GB+ VRAM (user has RTX 3050 6GB — insufficient)
- Training data (financial conversations, Indian market context)
- inference pipeline

**Decision:** Keep FinBERT as the primary model. Add multi-source headlines (MoneyControl, Economic Times) as a future enhancement. Note FinGPT as a documentation upgrade path.

### MF data source: yfinance vs AMFI

| Source | Cost | Reliability | Coverage |
|--------|------|-------------|----------|
| yfinance (current) | Free | Delayed, gaps | Has NAV for most Indian MFs |
| AMFI API | Free | Official, reliable | Requires API key registration |
| MFUtility | Free | Official | Complex SOAP API |

**Decision:** Use yfinance for NAV history (already works). Document AMFI as a production upgrade path. Keep it simple.

---

## Execution Plan: 5 Tasks

### Task 1: Stacked Meta-Learner
**Impact:** Critical — replaces naive equal-weight ensemble with learned weights
**Effort:** 4-5 hours
**Files:** `src/ensemble.py`, `tests/test_ensemble.py`

**Why:** The current ensemble uses equal weights (20% each). A meta-learner learns the optimal combination from out-of-sample data. This is the single highest-impact modeling improvement.

**Current state:**
```python
# src/ensemble.py — predict_ensemble()
dl_votes = d_l + d_g + d_t
dl_dir = 1 if dl_votes >= 2 else 0
ensemble_prob = dl_dir * 0.4 + xgb_p * 0.6  # HARDCODED
```

**Target:**
```python
# Train a LogisticRegression on model outputs
from sklearn.linear_model import LogisticRegression

meta_X = np.column_stack([lstm_probs, gru_probs, transformer_probs, xgb_probs, lgb_probs])
meta_model = LogisticRegression(C=1.0, max_iter=1000)
meta_model.fit(meta_X_train, y_train)
final_prob = meta_model.predict_proba(meta_X_test)[:, 1]
```

**Implementation steps:**

1. Add `train_meta_learner(meta_X, y)` function to `ensemble.py`
   - Input: (N, 5) array of base model probabilities
   - Output: fitted LogisticRegression
   - Use `Pipeline` with `StandardScaler` + `LogisticRegression`

2. Add `predict_with_metalearner(models, meta_model, data)` function
   - Runs each base model → collects probabilities → passes to meta-learner
   - Falls back to equal-weight if meta_model is None

3. Add `evaluate_metalearner(meta_X_test, y_test, meta_model, equal_weight_preds)`
   - Compare meta-learner accuracy vs equal-weight baseline
   - Return improvement percentage

4. Update `backtest_ensemble()` to optionally use meta-learner
   - Add `meta_model=None` parameter
   - When provided, use learned weights instead of hardcoded

5. Add persistence: `save_meta_model()` / `load_meta_model()`
   - Save as `models/{ticker}/meta_model.pkl`

**Tests:** `tests/test_ensemble.py`
- `test_meta_learner_trains_successfully` — fits on (100, 5) input
- `test_meta_learner_returns_probabilities` — output in [0, 1]
- `test_meta_learner_beats_equal_weight` — on synthetic data with known signal
- `test_meta_learner_fallback` — when meta_model=None, uses equal weight
- `test_meta_learner_save_load` — roundtrip to disk
- `test_meta_learner_handles_nan` — gracefully handles missing model outputs
- `test_meta_learner_coefficients_sum_to_one` — weights are interpretable

---

### Task 2: Regime-Conditional Routing
**Impact:** High — adapts ensemble to market conditions
**Effort:** 3-4 hours
**Files:** `src/ensemble.py`, `src/regime.py`, `tests/test_ensemble.py`

**Why:** Different models perform differently in different regimes. LSTM might be better in trending markets, XGBoost in mean-reverting. Routing gives each model its best environment.

**Current state:**
```python
# regime.py detects regime
# ensemble.py uses fixed weights regardless of regime
```

**Target:**
```python
REGIME_WEIGHTS = {
    "Bull":     {"lstm": 0.15, "gru": 0.15, "transformer": 0.2, "xgb": 0.25, "lgb": 0.25},
    "Bear":     {"lstm": 0.25, "gru": 0.25, "transformer": 0.15, "xgb": 0.15, "lgb": 0.2},
    "Sideways": {"lstm": 0.2, "gru": 0.2, "transformer": 0.2, "xgb": 0.2, "lgb": 0.2},
}
```

**Implementation steps:**

1. Add `REGIME_WEIGHTS` dict to `ensemble.py`
   - Bull: favor tree models (XGB/LGB capture momentum)
   - Bear: favor DL models (LSTM/GRU capture trend reversals)
   - Sideways: equal weight (no clear advantage)

2. Add `get_regime_weights(regime: str) -> dict` function
   - Returns weights for given regime
   - Falls back to equal weight for unknown regime

3. Modify `predict_ensemble()` to accept `regime` parameter
   - When regime is provided, use regime-specific weights
   - When regime is None, use default equal weights

4. Update `backtest_ensemble()` to pass regime through
   - Detect regime at each bar → route to appropriate weights

5. Add `evaluate_regime_routing()` to compare:
   - Equal-weight vs fixed-weight vs regime-routed performance

**Tests:** `tests/test_ensemble.py`
- `test_regime_weights_valid` — all regimes have 5 weights summing to 1.0
- `test_get_regime_weights_bull` — returns bull-specific weights
- `test_get_regime_weights_unknown_fallback` — falls back to equal weight
- `test_predict_with_regime` — uses regime-specific weights
- `test_regime_routing_improves_sharpe` — on synthetic data with regime shifts

---

### Task 3: Mutual Fund Tracker
**Impact:** High — integrates MF portfolio with stock signals
**Effort:** 5-6 hours
**Files:** New `src/mf_tracker.py`, `tests/test_mf_tracker.py`

**Why:** Many Indian investors hold both stocks and MFs. A unified view is essential. Current `holdings.py` only parses Zerodha CSVs — no live NAV tracking, no factor analysis.

**Current state:**
```python
# holdings.py: parse_holdings_csv(), compute_portfolio_stats()
# INDIAN_MF_MAP: 21 funds mapped to yfinance tickers
# No mf_tracker.py exists
```

**Implementation:**

```python
# src/mf_tracker.py

class MFTracker:
    """Mutual fund portfolio tracker with NAV history and factor analysis."""

    def __init__(self):
        self.holdings = {}  # ticker -> {"units": float, "avg_nav": float, "fund_name": str}
        self.nav_cache = {}  # ticker -> pd.Series of NAV history

    def add_holding(self, ticker: str, units: float, avg_nav: float, fund_name: str = ""):
        """Add or update a MF holding."""

    def fetch_nav_history(self, ticker: str, period: str = "2y") -> pd.Series:
        """Fetch NAV history from yfinance. Caches in memory."""

    def get_current_value(self, ticker: str) -> float:
        """Latest NAV × units."""

    def get_portfolio_value(self) -> float:
        """Total value across all holdings."""

    def compute_xirr(self, ticker: str = None) -> float:
        """XIRR for a specific fund or entire portfolio."""

    def compute_factor_exposures(self, ticker: str) -> dict:
        """Estimate value/momentum/quality factor loadings from NAV returns."""

    def get_top_holdings(self, n: int = 5) -> list:
        """Top N holdings by value."""

    def get_allocation_breakdown(self) -> dict:
        """Category-level allocation (Index, Debt, ELSS, etc.)."""

    def detect_concentration_risk(self, threshold: float = 0.3) -> list:
        """Flag holdings exceeding threshold."""

    def compare_to_benchmark(self, ticker: str, benchmark: str = "^NSEI") -> dict:
        """Compare fund returns vs Nifty 50."""

    def save_state(self, path: str = "mf_state.json"):
        """Persist holdings + cached NAVs."""

    def load_state(self, path: str = "mf_state.json") -> bool:
        """Load from disk."""
```

**Integration with existing code:**
- Import `INDIAN_MF_MAP` from `holdings.py` for fund→ticker mapping
- Use `compute_xirr()` from `holdings.py` (already exists)
- Add `tab_mf_tracker()` to app.py (17th tab)

**Tests:** `tests/test_mf_tracker.py`
- `test_add_holding` — stores correctly
- `test_get_portfolio_value` — sums all holdings
- `test_compute_xirr_positive` — positive return → positive XIRR
- `test_compute_xirr_negative` — negative return → negative XIRR
- `test_allocation_breakdown` — sums to 100%
- `test_concentration_risk_detects` — flags >30% weight
- `test_save_load_state` — roundtrip to disk
- `test_fetch_nav_returns_series` — yfinance returns pd.Series
- `test_compare_to_benchmark` — returns dict with all keys

---

### Task 4: Expand Universe + Batch Training
**Impact:** Medium — more stocks = more opportunities
**Effort:** 2-3 hours
**Files:** `src/trainer.py`, `run_pipeline.py`, `tests/test_trainer.py`

**Why:** Only 3 of 20 stocks are trained. Batch training covers the full universe.

**Current state:**
```python
# NSE_STOCKS = 20 stocks (in data_fetcher.py)
# Training: manual "Train" click per stock in UI
# No batch training command
```

**Implementation:**

1. Add `batch_train(tickers: list, config: TrainConfig = None)` to `trainer.py`
   - Iterates through tickers, trains each, logs results
   - Skips already-trained tickers (unless force=True)
   - Returns summary: {success: [], failed: [], skipped: []}

2. Add `--train-all` flag to `run_pipeline.py`
   - Trains all 20 stocks in NSE_STOCKS
   - Reports per-stock status

3. Add `--train` flag with specific tickers
   - `python run_pipeline.py --train RELIANCE.NS TCS.NS`

4. Update pipeline to support batch training stage
   - PipelineConfig gets `train_tickers` list
   - Pipeline stage: "train" runs batch_train

**Tests:** `tests/test_trainer.py`
- `test_batch_train_skips_existing` — doesn't retrain trained stocks
- `test_batch_train_force` — retrains when force=True
- `test_batch_train_returns_summary` — has success/failed/skipped keys
- `test_batch_train_handles_failure` — continues on individual failure

---

### Task 5: Fundamental Ranking Factors
**Impact:** Medium — better stock selection with fundamentals
**Effort:** 3-4 hours
**Files:** `src/ranking.py`, `tests/test_ranking.py`

**Why:** Current ranking uses only technical factors. Fundamental factors (P/E, P/B, ROE) improve long-term stock selection.

**Current state:**
```python
# ranking.py: momentum, volatility, volume, technical scores
# No fundamental data (P/E, P/B, ROE, dividend yield)
```

**Implementation:**

1. Add `fetch_fundamentals(ticker: str) -> dict` to ranking.py
   - Uses yfinance `.info` to get: trailingPE, priceToBook, returnOnEquity, dividendYield, marketCap, debtToEquity
   - Caches results (fundamentals change slowly)

2. Add `fundamental_score(fundamentals: dict) -> float`
   - P/E: lower is better (0-30 range normalized to 0-100)
   - P/B: lower is better
   - ROE: higher is better
   - Dividend yield: higher is better
   - Combined: weighted average

3. Update `rank_stocks()` to include `fundamental_weight` parameter
   - Default 0 (backward compatible)
   - When >0, adds fundamental score to composite

4. Update `tab_ranking()` in app.py
   - Add "Include Fundamentals" checkbox
   - When enabled, fetches fundamentals for displayed stocks

**Tests:** `tests/test_ranking.py`
- `test_fetch_fundamentals_returns_dict` — has expected keys
- `test_fundamental_score_in_range` — between 0 and 100
- `test_fundamental_score_high_roe` — high ROE → high score
- `test_rank_stocks_with_fundamentals` — incorporates fundamental factor
- `test_rank_stocks_backward_compatible` — default weight=0 unchanged

---

## Execution Order

```
1. Task 1: Stacked Meta-Learner (4-5 hrs)    ← CRITICAL, foundation for Task 2
2. Task 2: Regime-Conditional Routing (3-4 hrs) ← depends on Task 1 (meta-learner)
3. Task 3: MF Tracker (5-6 hrs)               ← independent, can parallel with 1-2
4. Task 4: Expand Universe (2-3 hrs)           ← independent, can parallel with 1-3
5. Task 5: Fundamental Ranking (3-4 hrs)       ← independent, can parallel with 1-4
```

**Parallelizable:**
- Tasks 3, 4, 5 are independent of each other and of Tasks 1-2
- Task 2 requires Task 1 (meta-learner must exist before routing)

**Total estimated time:** 17-22 hours

---

## Exit Criteria Checklist

After all 5 tasks:

- [x] Stacked meta-learner trained and evaluated on all 3 trained stocks
- [x] Meta-learner shows measurable improvement over equal-weight baseline
- [x] Regime-conditional routing implemented and tested
- [x] MF tracker with NAV history, XIRR, factor exposures
- [x] All 20 stocks batch-trainable
- [x] Fundamental factors in ranking (P/E, P/B, ROE, dividend yield)
- [x] All new modules have tests passing
- [x] Lint clean on all new/modified files
- [x] Open-source license (Apache 2.0) and CONTRIBUTING.md added

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/mf_tracker.py` | MF portfolio tracker: NAV history, XIRR, factor exposures, allocation |
| `tests/test_mf_tracker.py` | 9 tests for MF tracker |
| `LICENSE` | MIT license |
| `CONTRIBUTING.md` | Contribution guidelines |

## Files to Modify

| File | Change |
|------|--------|
| `src/ensemble.py` | Add `train_meta_learner()`, `REGIME_WEIGHTS`, regime-aware `predict_ensemble()` |
| `src/ranking.py` | Add `fetch_fundamentals()`, `fundamental_score()`, update `rank_stocks()` |
| `src/trainer.py` | Add `batch_train()` for multi-ticker training |
| `run_pipeline.py` | Add `--train-all` and `--train` flags |
| `app.py` | Add "MF Tracker" tab (17th tab), update ranking tab for fundamentals |
| `tests/test_ensemble.py` | 7 new tests for meta-learner + regime routing |
| `tests/test_ranking.py` | 5 new tests for fundamental factors |
| `tests/test_trainer.py` | 4 new tests for batch training |

---

## Critical Context

- **User's venv:** `C:\Users\uttam\venv` (Python 3.14, torch 2.10+cu130, scipy installed)
- **User's project:** `C:\Users\uttam\development\stomar`
- **Run tests:** `$env:PYTHONPATH = "C:\Users\uttam\development\stomar"; python -m pytest tests/ -v`
- **Run linter:** `python -m ruff check src/ tests/ --output-format=concise`
- **Current test count:** 424 tests passing
- **Trained stocks:** 3 of 20 (RELIANCE, TCS, INFY likely)
- **User constraint:** Everything must be free, must work on Windows
- **Key insight:** This stage is about INTELLIGENCE — making the ensemble smarter and coverage broader
- **Meta-learner data:** ~250 trading days per walk-forward window — use simple model (LogisticRegression), not neural net
- **FinGPT note:** Documented as upgrade path but not feasible on user's hardware (6GB VRAM)
- **Paper trading clock:** Started in Stage 4 — continue running for 3+ months before real capital

---

## Appendix: Stage 4 (Completed)

Stage 4 built the execution layer. 89 new tests added, 424 total.

| Component | File | Status |
|-----------|------|--------|
| Event-driven engine | `src/engine.py` | ✅ |
| Risk controls | `src/risk_controls.py` | ✅ |
| Paper trader | `src/paper_trader.py` | ✅ |
| Execution quality | `src/execution_quality.py` | ✅ |
| Fill probability | `src/engine.py` | ✅ |
| State persistence | `src/paper_trader.py` | ✅ |
| Paper trading tab | `app.py` | ✅ |
| Pipeline --paper flag | `run_pipeline.py` | ✅ |

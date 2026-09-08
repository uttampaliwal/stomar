"""F1 stale-combiner quarantine (safety, not a gate retry).

Quarantined 2026-09-08 after holdout measurement (scripts/refresh_meta.py,
data/edge_gate/meta_refresh_20260907_183616.json):

- meta_BAJFINANCE_NS: holdout 0.511 < equal-weight 0.573 (worse than nothing)
- meta_MARUTI_NS: holdout 0.524 == equal-weight 0.524 (no value, stale)

Production (orchestrator._run_ensemble) falls back to the active-average
ensemble when no meta file exists — strictly better by measurement.
Restore by re-running scripts/refresh_meta.py after signal changes.
"""

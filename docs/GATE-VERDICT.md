# StoMar — Gate Verdict (F3): alpha track closed, infra track open

> This file records outcomes. The contract (`docs/MASTER-PLAN.md`,
> `docs/Go-Live-Gate.md`) is unchanged. Live real-money trading: **NO-GO**.

## Evidence chain (all pooled, net of full NSE costs, purged/embargoed rows)

| Gate | Window | acc_ens | vs logistic | ret_ens vs buy-hold | McNemar | Verdict |
|---|---|---|---|---|---|---|
| O1b baseline | last 30% (n=4966) | 0.590 | 0.590 < ~0.61 | −0.074 < −0.018 | 0.36 | FAIL |
| F1 re-run (11/20 metas refit, prod path) | same | 0.606 | 0.606 < ~0.614 | −0.081 < −0.018 | 0.059 | FAIL |
| F2 extended (eval 50%, n=8290) | last 50% | 0.701 | behind in bull/bear/side | −0.161 < −0.010 | 0.0245 | FAIL, no winning regime |
| F2b threshold 0.6 (one exception) | last 30% | 0.600 | behind | −0.091 < −0.015 | 0.0112 | FAIL |

Reports: `data/edge_gate/report_*.json` (local, gitignored).

## What was learned (measured, not speculated)

1. Directional signal is real (59–70% vs ~50% baselines, p≈0) — not a coin flip.
2. The combiner is at parity with a plain refit logistic (gap ≤0.014) — no
   stacking alpha left to squeeze.
3. The binding constraint is strategy economics: 0.31 flips/bar at the 0.5
   threshold converts direction into churn (e.g. INFY 78% acc, −58% net on
   215 flips). Threshold 0.6 cut turnover but did not fix economics.
4. Train/serve parity fixed en route (16 SOTA factors were zero-filled live);
   two worse-than-nothing stale metas quarantined (`models/_quarantine/`).

## F3 consequences

- **Closed**: O5/O6 model/optimizer spend aimed at alpha; any live-money path
  stays NO-GO under `docs/Go-Live-Gate.md`.
- **Open**: execution/portfolio/research infra (cost engine, risk controls,
  backtesting, ledger, wealth tracking); paper-trading research use;
  60-day paper clock may continue as measurement, not as graduation.
- **Reopening** the alpha track requires a new hypothesis doc + new gate —
  never a quiet threshold tweak. The bar is the same frozen gate.

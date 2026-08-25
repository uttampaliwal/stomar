"""Authoritative daily-bar simulation core — the single source of truth for
signal-to-position semantics and transaction costs.

Every performance number this project reports (trainer OOS metrics, pipeline
evaluate gate, research notebooks) must derive from this module so results
are comparable.

Convention (one way, no exceptions):

1. A signal at bar ``t`` uses only data up to ``t`` and predicts the
   ``t -> t+1`` move.
2. The position implied by that signal is entered at bar ``t``'s close and
   earns the close-to-close forward return of ``t -> t+1``.
3. Every position CHANGE pays real NSE costs: slippage plus the full
   ``calculate_nse_costs`` stack (brokerage, STT on both sides, exchange
   charges, SEBI fees, stamp duty, GST), expressed as effective rates on
   traded value. Holding an unchanged position is free beyond its market
   risk; entering pays the buy rate, exiting pays the sell rate, and a
   long<->short flip crosses zero so it correctly pays both legs.

Costless "paper" Sharpe numbers are not comparable to costed ones; anything
costless is a signal-quality diagnostic at best.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from src.core.constants import SLIPPAGE_RATE, calculate_nse_costs


@lru_cache(maxsize=1)
def nse_cost_rates() -> dict:
    """Effective one-side cost rates as fractions of traded value.

    Returns:
        Dict with ``buy`` and ``sell`` rates, each = NSE effective rate
        (brokerage + STT + exchange + SEFI + stamp duty + GST) + slippage.
        Sell side is cheaper (no stamp duty), mirroring real NSE delivery
        economics.
    """
    # Rates are proportional to value, so price/quantity are arbitrary.
    buy = calculate_nse_costs(1000.0, 100, "buy")["effective_rate"]
    sell = calculate_nse_costs(1000.0, 100, "sell")["effective_rate"]
    return {
        "buy": round(buy + SLIPPAGE_RATE, 6),
        "sell": round(sell + SLIPPAGE_RATE, 6),
    }


def positions_from_predictions(
    y_pred,
    strategy: str = "directional",
    exposure: float = 1.0,
) -> np.ndarray:
    """Map model predictions to per-bar positions under the shared convention.

    Args:
        y_pred: Predicted probability of an up move at each bar.
        strategy: "directional" — long ``exposure`` when p > 0.5 else flat;
            "signed" — position ``(2p - 1) * exposure`` in [-exposure, exposure].
        exposure: Fraction of equity deployed at full conviction.

    Returns:
        Float array of positions aligned with ``y_pred``.
    """
    y_pred = np.asarray(y_pred, dtype=float)
    if strategy == "signed":
        return (2.0 * np.clip(y_pred, 0.0, 1.0) - 1.0) * float(exposure)
    return np.where(y_pred > 0.5, float(exposure), 0.0)


def strategy_return_series(positions, forward_returns) -> np.ndarray:
    """Cost-adjusted strategy returns for a position path.

    Args:
        positions: Position held over each bar's t -> t+1 move (fraction of
            equity). Positive = long, negative = short.
        forward_returns: Market return over the same t -> t+1 interval.

    Returns:
        Strategy return per bar: ``pos * fwd`` minus turnover costs charged
        on every position change at the NSE buy/sell effective rates.
    """
    positions = np.asarray(positions, dtype=float)
    fwd = np.asarray(forward_returns, dtype=float)
    if positions.shape != fwd.shape:
        raise ValueError(
            f"positions shape {positions.shape} != forward_returns shape {fwd.shape}"
        )

    rates = nse_cost_rates()
    prev = np.concatenate(([0.0], positions[:-1]))
    delta = positions - prev
    turnover_cost = np.where(delta > 0, delta * rates["buy"], -delta * rates["sell"])
    return positions * fwd - turnover_cost

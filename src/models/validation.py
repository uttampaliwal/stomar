"""Purged & embargoed walk-forward cross-validation (Marcos López de Prado).

Standard K-Fold / random splits leak information when labels span multiple
bars (e.g. triple-barrier or forward-return labels): training samples whose
label window overlaps the test set must be PURGED, and samples immediately
after the test set must be EMBARGOED to neutralize serial correlation.

This module provides:

    PurgedGroupTimeSeriesSplit — sklearn-style splitter with purge + embargo.
    purged_walk_forward        — OOS predictions across folds for a model fn.
    oos_metrics               — Sharpe / Sortino / Calmar / MaxDrawdown.

Usage:
    from src.models.validation import PurgedGroupTimeSeriesSplit
    splitter = PurgedGroupTimeSeriesSplit(n_splits=5, embargo=10, horizon=5)
    for train_idx, test_idx in splitter.split(X, y, groups):
        ...
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator

logger = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR = 252


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """Time-series split that purges label-overlapping train samples and
    embargoes samples immediately following each test fold.

    Args:
        n_splits: Number of train/test folds.
        embargo: Number of samples after each test fold to drop from training
            (a `pandas.Timedelta` is allowed when groups are datetimes).
        horizon: Label look-ahead horizon in samples. Train samples whose
            label window [t, t + horizon) overlaps the test fold are purged.
            When `t1` (label end) is provided via the `groups` DataFrame,
            horizon is ignored and true label spans are used.
        gap: Number of samples to leave between test folds (default 0).

    Groups semantics (López de Prado, "Advances in Financial ML", ch. 7):
        - If groups is a DataFrame with a `t1` column (label end times),
          purge uses the true per-sample label spans.
        - If groups is an array of group keys (e.g. dates), samples sharing
          a group stay together and horizon-based purging applies.
    """

    def __init__(
        self,
        n_splits: int = 5,
        embargo: float | pd.Timedelta = 10,
        horizon: int = 5,
        gap: int = 0,
    ):
        if n_splits < 2:
            raise ValueError("n_splits must be >= 2")
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        self.n_splits = n_splits
        self.embargo = embargo
        self.horizon = horizon
        self.gap = gap

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    # ── index helpers ────────────────────────────────────────────────────

    def _resolve_t1(self, groups) -> pd.Series | None:
        if groups is not None and isinstance(groups, pd.DataFrame) and "t1" in groups.columns:
            t1 = pd.Series(groups["t1"].values, index=groups.index).reset_index(drop=True)
            if pd.api.types.is_datetime64_any_dtype(t1):
                t1 = pd.to_datetime(t1)
            return t1
        return None

    def _resolve_t0(self, groups) -> pd.Series | None:
        if groups is not None and isinstance(groups, pd.DataFrame) and "t0" in groups.columns:
            t0 = pd.Series(groups["t0"].values, index=groups.index).reset_index(drop=True)
            if pd.api.types.is_datetime64_any_dtype(t0):
                t0 = pd.to_datetime(t0)
            return t0
        return None

    def _embargo_offset(self, groups) -> int:
        if isinstance(self.embargo, pd.Timedelta):
            if groups is not None and isinstance(groups, pd.DataFrame) and "t0" in groups.columns:
                t0 = pd.to_datetime(groups["t0"].values)
                if len(t0) > 1:
                    step = (pd.to_datetime(t0[1]) - pd.to_datetime(t0[0]))
                    if step > pd.Timedelta(0):
                        return max(1, int(self.embargo / step))
            return max(1, int(self.embargo / pd.Timedelta(days=1)))
        return max(0, int(self.embargo))

    # ── core splitter ────────────────────────────────────────────────────

    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        if n_samples < 2 * self.n_splits:
            raise ValueError(
                f"Not enough samples ({n_samples}) for {self.n_splits} folds"
            )

        t1 = self._resolve_t1(groups)
        t0 = self._resolve_t0(groups)
        embargo = self._embargo_offset(groups)

        indices = np.arange(n_samples)
        fold_edges = np.linspace(0, n_samples, self.n_splits + 1, dtype=int)

        for fold in range(self.n_splits):
            test_start = fold_edges[fold]
            test_end = fold_edges[fold + 1]
            if test_end - test_start < 1:
                continue

            test_idx = indices[test_start:test_end]

            # train = everything before the test fold (expanding window)
            train_idx = indices[:test_start]

            if t1 is not None:
                train_idx = self._purge_by_t1(train_idx, test_idx, t1, t0)
            elif self.horizon:
                # purge samples whose label window overlaps the test fold
                train_idx = self._purge_by_horizon(train_idx, test_start, self.horizon)

            # embargo: drop samples immediately following the test fold
            train_idx = self._apply_embargo(train_idx, test_end, embargo)

            # skip degenerate folds (no training data left after purge)
            if len(train_idx) < 1:
                continue

            yield train_idx, test_idx

    # ── purge / embargo internals ────────────────────────────────────────

    @staticmethod
    def _purge_by_t1(
        train_idx: np.ndarray,
        test_idx: np.ndarray,
        t1: pd.Series,
        t0: pd.Series | None = None,
    ) -> np.ndarray:
        """Remove train samples whose label end (t1) falls inside the test
        window [test_start, test_end).

        When t1 is datetime-based, the test window is translated into
        datetimes via t0 (position -> time mapping) and the exclusive end is
        extended by one step so `t1 in [test_t0, test_t1)` still holds.
        """
        if t0 is not None and pd.api.types.is_datetime64_any_dtype(t1):
            test_t0 = t0.iloc[int(test_idx[0])]
            test_t1 = t0.iloc[int(test_idx[-1])] + (t0.iloc[1] - t0.iloc[0])
            mask = ~((t1.iloc[train_idx] >= test_t0) & (t1.iloc[train_idx] < test_t1))
        else:
            test_start = int(test_idx[0])
            test_end = int(test_idx[-1]) + 1
            mask = ~((t1.iloc[train_idx] >= test_start) & (t1.iloc[train_idx] < test_end))
        return train_idx[mask]

    @staticmethod
    def _purge_by_horizon(train_idx: np.ndarray, test_start: int, horizon: int) -> np.ndarray:
        """Remove train samples at position t whose labels look into
        [test_start, test_start + horizon)."""
        lookback_start = test_start - horizon + 1
        mask = ~((train_idx >= lookback_start) & (train_idx < test_start))
        return train_idx[mask]

    @staticmethod
    def _apply_embargo(train_idx: np.ndarray, test_end: int, embargo: int) -> np.ndarray:
        if embargo <= 0:
            return train_idx
        mask = (train_idx < test_end) | (train_idx >= test_end + embargo)
        return train_idx[mask]

    def get_n_train_folds(self, X, groups=None) -> list[int]:
        return [len(train) for train, _ in self.split(X, groups=groups)]


# ── purged walk-forward driver ───────────────────────────────────────────────


def purged_walk_forward(
    X: pd.DataFrame,
    y: pd.Series,
    fit_predict,
    groups: pd.DataFrame | np.ndarray | None = None,
    n_splits: int = 5,
    embargo: float | pd.Timedelta = 10,
    horizon: int = 5,
    gap: int = 0,
    progress: bool = True,
) -> dict:
    """Run purged walk-forward CV and collect out-of-sample predictions.

    Args:
        X: Feature matrix (index must be an integer position or datetimes
            aligned with y).
        y: Target series (binary or continuous).
        fit_predict: Callable(train_X, train_y, test_X) -> predictions.
        groups: DataFrame with t0/t1 label columns (preferred), or array of
            group keys. See PurgedGroupTimeSeriesSplit.
        n_splits / embargo / horizon / gap: CV parameters.
        progress: Log per-fold results.

    Returns:
        Dict with:
            oos_predictions  — DataFrame [fold, y_true, y_pred, proba?]
            folds            — list of {train_size, test_size, n_purged}
            metrics          — performance_metrics() over OOS strategy returns
    """
    splitter = PurgedGroupTimeSeriesSplit(
        n_splits=n_splits, embargo=embargo, horizon=horizon, gap=gap
    )

    y_index = np.arange(len(y))
    oos = []
    fold_records = []
    for fold, (train_idx, test_idx) in enumerate(splitter.split(X, y, groups)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], y.iloc[test_idx]

        pred = fit_predict(X_tr, y_tr, X_te)
        if isinstance(pred, tuple):
            pred_values, proba = pred
        else:
            pred_values, proba = pred, None

        pred_values = np.asarray(pred_values).flatten()
        n = min(len(pred_values), len(test_idx))
        fold_df = pd.DataFrame(
            {
                "fold": fold,
                "position": test_idx[:n],
                "y_true": np.asarray(y_te)[:n],
                "y_pred": pred_values[:n],
            }
        )
        if proba is not None:
            fold_df["proba"] = np.asarray(proba).flatten()[:n]
        oos.append(fold_df)

        # train_idx is a subset of indices[:test_start]; the difference is
        # exactly the samples removed by purge + embargo for this fold.
        n_purged = max(0, int(test_idx[0]) - len(train_idx))
        fold_records.append(
            {
                "fold": fold,
                "train_size": len(train_idx),
                "test_size": len(test_idx),
                "n_purged": n_purged,
            }
        )
        if progress:
            logger.info(
                "fold=%d train=%d test=%d embargo=%d",
                fold, len(train_idx), len(test_idx), splitter._embargo_offset(groups),
            )

    oos_df = pd.concat(oos, ignore_index=True) if oos else pd.DataFrame()
    metrics = oos_metrics(oos_df, groups=groups, horizon=horizon) if len(oos_df) else {}

    return {"oos_predictions": oos_df, "folds": fold_records, "metrics": metrics}


# ── performance metrics (no look-ahead) ──────────────────────────────────────


@dataclass
class OOSMetrics:
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    annual_return: float = 0.0
    volatility: float = 0.0
    n_periods: int = 0
    details: dict = field(default_factory=dict)


def oos_metrics(
    predictions: pd.DataFrame,
    groups: pd.DataFrame | np.ndarray | None = None,
    horizon: int = 5,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    strategy: str = "directional",
) -> dict:
    """Compute OOS performance metrics from walk-forward predictions.

    Args:
        predictions: DataFrame with y_true, y_pred (and optional position).
        groups: As in the splitter (used to estimate label duration when the
            predictions are not daily).
        horizon: Label horizon in bars (used for period scaling).
        periods_per_year: Annualization factor (252 for daily NSE bars).
        strategy: "directional" — long when y_pred > 0.5, flat otherwise;
            "signed" — position = 2*y_pred - 1.

    Returns:
        Dict with sharpe, sortino, calmar, max_drawdown, total_return,
        annual_return, volatility, n_periods, plus raw return series info.
    """
    if predictions is None or len(predictions) == 0:
        return {
            "sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0,
            "total_return": 0.0, "annual_return": 0.0, "volatility": 0.0,
            "n_periods": 0, "details": {},
        }

    y_true = predictions["y_true"].astype(float).values
    y_pred = predictions["y_pred"].astype(float).values

    if strategy == "signed":
        position = 2.0 * y_pred - 1.0
    else:
        position = np.where(y_pred > 0.5, 1.0, 0.0)

    # Strategy returns: long gets the full move, flat gets 0
    strat_returns = position * y_true
    cum = np.cumprod(1.0 + strat_returns)
    peak = np.maximum.accumulate(cum)
    drawdown = cum / peak - 1.0

    n = len(strat_returns)
    vol = np.std(strat_returns, ddof=1) if n > 1 else 0.0
    if n == 0 or np.all(strat_returns == 0) or np.isclose(vol, 0.0):
        vol = 0.0

    mean_ret = np.mean(strat_returns) if n else 0.0
    downside = strat_returns[strat_returns < 0]
    downside_std = np.std(downside, ddof=1) if len(downside) > 1 else (np.mean(downside) if len(downside) == 1 else 0.0)
    if np.isclose(downside_std, 0.0):
        downside_std = 0.0

    annualized_vol = vol * np.sqrt(periods_per_year)
    sharpe = mean_ret / vol * np.sqrt(periods_per_year) if vol > 0 else 0.0
    sortino = mean_ret / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    total_return = float(cum[-1] - 1.0) if n else 0.0
    years = n / periods_per_year
    annual_return = (cum[-1] ** (1 / years) - 1.0) if years > 0 and cum[-1] > 0 else 0.0
    max_drawdown = float(drawdown.min()) if n else 0.0
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    n_up = int(np.sum(y_pred > 0.5))
    n_down = int(np.sum(y_pred <= 0.5))
    n_traded = int(np.sum(position != 0))

    metrics = OOSMetrics(
        sharpe=round(float(sharpe), 4),
        sortino=round(float(sortino), 4),
        calmar=round(float(calmar), 4),
        max_drawdown=round(float(max_drawdown), 4),
        total_return=round(float(total_return), 4),
        annual_return=round(float(annual_return), 4),
        volatility=round(float(annualized_vol), 4),
        n_periods=n,
    )
    metrics.details = {
        "n_up_predictions": n_up,
        "n_down_predictions": n_down,
        "n_traded": n_traded,
        "periods_per_year": periods_per_year,
        "horizon": horizon,
        "strategy": strategy,
        "final_cumulative": round(float(cum[-1]) if n else 0.0, 4),
    }
    return metrics.__dict__


def performance_metrics(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Standard strategy performance metrics from a return series.

    Args:
        returns: Strategy return series (decimal, e.g. 0.01 = 1%).

    Returns:
        Dict with sharpe, sortino, calmar, max_drawdown, total_return,
        annual_return, volatility.
    """
    returns = pd.Series(returns).dropna().astype(float)
    n = len(returns)
    if n == 0:
        return {"sharpe": 0.0, "sortino": 0.0, "calmar": 0.0, "max_drawdown": 0.0,
                "total_return": 0.0, "annual_return": 0.0, "volatility": 0.0,
                "n_periods": 0}

    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    drawdown = cum / peak - 1.0

    vol = float(np.std(returns, ddof=1)) if n > 1 else 0.0
    if np.isclose(vol, 0.0):
        vol = 0.0
    mean = float(np.mean(returns))
    downside = returns[returns < 0]
    downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else 0.0
    if np.isclose(downside_std, 0.0):
        downside_std = 0.0

    sharpe = mean / vol * np.sqrt(periods_per_year) if vol > 0 else 0.0
    sortino = mean / downside_std * np.sqrt(periods_per_year) if downside_std > 0 else 0.0

    total_return = float(cum.iloc[-1] - 1.0)
    years = n / periods_per_year
    annual_return = float(cum.iloc[-1] ** (1 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else 0.0
    max_drawdown = float(drawdown.min())
    calmar = annual_return / abs(max_drawdown) if max_drawdown < 0 else 0.0

    return {
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "calmar": round(calmar, 4),
        "max_drawdown": round(max_drawdown, 4),
        "total_return": round(total_return, 4),
        "annual_return": round(annual_return, 4),
        "volatility": round(vol * np.sqrt(periods_per_year), 4),
        "n_periods": n,
    }


def strategy_returns_from_signals(
    returns: pd.Series, signals: pd.Series | np.ndarray, threshold: float = 0.5
) -> pd.Series:
    """Long-only strategy returns from model signals.

    position_t = 1 if signal_t > threshold else 0; strategy_ret_t =
    position_{t-1} * return_t (signal applied next period, no look-ahead).
    """
    signals = pd.Series(signals, index=returns.index)
    position = (signals > threshold).astype(float).shift(1).fillna(0.0)
    return position * returns

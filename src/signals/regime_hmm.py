"""Hidden Markov / Gaussian Mixture market regime engine.

Identifies 3 market regimes from daily OHLCV data:

    * High Volatility Bear Market   — panic / crash states
    * Trending Bull Market          — strong positive drift
    * Low Volatility Ranging Market — sideways chop

A Gaussian Mixture Model (sklearn) is fit on standardized daily features:
log returns, 10-day realized volatility and ATR-as-% of price. The GMM is the
observational-emission model of an HMM without transition dynamics, which is
sufficient for regime *classification*; the module exposes the same interface
an hmmlearn-based engine would (state probabilities + risk scaling), so an
HMM can be swapped in without touching callers.

The component -> regime label mapping is deterministic: the highest-volatility
state is the bear regime, the state with the strongest positive mean return
among the rest is the bull regime, and the residual is the ranging regime.

Outputs:
    detect_regime(ohlc)  -> {regime, regime_key, probabilities, risk, ...}
    get_risk_scaling(probs) -> probability-weighted {leverage, position_scale,
                               max_risk_per_trade, hedge} factors
    state_sequence(ohlc) -> per-day regime keys (for regime-conditional
                            rolling performance of ensemble models)

Usage:
    from src.signals.regime_hmm import detect_regime
    info = detect_regime(df)   # df with open/high/low/close/volume
    print(info["regime"], info["probabilities"], info["risk"]["leverage"])
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

N_REGIMES = 3
WINDOW = 504  # ~2 trading years

# Human-readable regime labels (order-independent; keys are stable)
REGIME_LABELS = {
    "bear": "High Volatility Bear Market",
    "bull": "Trending Bull Market",
    "ranging": "Low Volatility Ranging/Sideways Market",
}

# Dynamic risk-scaling factors per regime (leverage multiplier, position
# scale, max risk per trade, and whether hedging is advised).
RISK_SCALING = {
    "bear": {
        "leverage": 0.5,
        "position_scale": 0.5,
        "max_risk_per_trade": 0.01,
        "hedge": True,
        "note": "Scale leverage to 0.5x, cut position sizes, prefer hedges",
    },
    "bull": {
        "leverage": 1.0,
        "position_scale": 1.0,
        "max_risk_per_trade": 0.02,
        "hedge": False,
        "note": "Full leverage, trend-follow the up-drift",
    },
    "ranging": {
        "leverage": 0.75,
        "position_scale": 0.75,
        "max_risk_per_trade": 0.015,
        "hedge": False,
        "note": "Reduced leverage, mean-reversion friendly",
    },
}

# Tiny TTL cache so repeated endpoint calls don't refit the GMM
_cache: dict[str, tuple[float, dict]] = {}
_CACHE_TTL = 300  # 5 minutes


def _features(ohlc: pd.DataFrame, window: int = WINDOW) -> pd.DataFrame:
    """Daily regime features: log return, realized vol, ATR % of price."""
    if ohlc is None or len(ohlc) < 30:
        raise ValueError("regime_hmm requires >= 30 rows of OHLCV data")

    out = pd.DataFrame(index=ohlc.index)
    close = ohlc["close"].astype(float)

    log_ret = np.log(close / close.shift(1))
    out["log_return"] = log_ret

    # Realized volatility: 10-day std of log returns (annualized in spirit)
    out["realized_vol"] = log_ret.rolling(10).std()

    # ATR as % of close (Wilder's ATR via ewm, matching ta conventions)
    high, low = ohlc["high"].astype(float), ohlc["low"].astype(float)
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 10, adjust=False).mean()
    out["atr_pct"] = atr / close

    return out.tail(window).dropna()


def _fit_gmm(features: pd.DataFrame, seed: int = 42) -> tuple[GaussianMixture, StandardScaler, np.ndarray]:
    """Fit a 3-state GMM on standardized features.

    Returns (gmm, scaler, standardized_matrix).
    """
    scaler = StandardScaler()
    X = scaler.fit_transform(features.values)
    gmm = GaussianMixture(
        n_components=N_REGIMES,
        covariance_type="diag",
        random_state=seed,
        max_iter=300,
        n_init=4,
    )
    gmm.fit(X)
    return gmm, scaler, X


def _label_states(
    gmm: GaussianMixture,
    scaler: StandardScaler,
    features: pd.DataFrame,
    X_scaled: np.ndarray,
) -> dict[int, str]:
    """Map GMM component indices to regime keys deterministically.

    Heuristic: the highest-volatility state is the bear regime; among the
    remaining states the one with the strongest positive mean log return is
    the bull regime; the residual is the ranging regime.
    """
    # Invert standardization to get original-scale means/std per component
    means = gmm.means_.copy()
    covs = gmm.covariances_
    means[:, 0] = means[:, 0] * scaler.scale_[0] + scaler.mean_[0]  # log_return
    means[:, 1] = means[:, 1] * scaler.scale_[1] + scaler.mean_[1]  # realized_vol
    means[:, 2] = means[:, 2] * scaler.scale_[2] + scaler.mean_[2]  # atr_pct

    vols = np.zeros(gmm.n_components)
    for i in range(gmm.n_components):
        c = covs[i] if covs[i].ndim == 2 else np.diag(covs[i])
        vols[i] = np.sqrt(c[1, 1]) * scaler.scale_[1]  # realized-vol std

    # Empirical per-component stats (robust to covariance types)
    labels = gmm.predict(X_scaled)
    for i in range(gmm.n_components):
        mask = labels == i
        if mask.sum() >= 2:
            means[i, 0] = float(np.mean(features.iloc[mask, 0]))
            vols[i] = float(np.std(features.iloc[mask, 1]))

    order = np.argsort(vols)[::-1]  # highest vol first
    mapping: dict[int, str] = {}
    mapping[int(order[0])] = "bear"  # most volatile state = bear

    rest = order[1:]
    if len(rest) >= 1:
        # bull = strongest mean log return among the rest
        rets = [means[int(i), 0] for i in rest]
        bull_idx = rest[int(np.argmax(rets))]
        mapping[int(bull_idx)] = "bull"
    for i in rest:
        if int(i) != int(bull_idx):
            mapping[int(i)] = "ranging"

    return mapping


def fit_regime_model(ohlc: pd.DataFrame, window: int = WINDOW, seed: int = 42) -> dict:
    """Fit the regime GMM and return the labeled model bundle."""
    features = _features(ohlc, window=window)
    gmm, scaler, X_scaled = _fit_gmm(features, seed=seed)
    mapping = _label_states(gmm, scaler, features, X_scaled)
    return {
        "gmm": gmm,
        "scaler": scaler,
        "features": features,
        "X_scaled": X_scaled,
        "state_map": mapping,
        "n_samples": len(features),
        "labels": REGIME_LABELS,
    }


def _probabilities(model: dict, features: pd.DataFrame) -> dict[str, float]:
    gmm = model["gmm"]
    mapping = model["state_map"]
    probs = gmm.predict_proba(model["X_scaled"])[-1]
    out = {REGIME_LABELS[mapping[i]]: float(probs[i]) for i in range(gmm.n_components)}
    return out


def get_risk_scaling(probabilities: dict[str, float]) -> dict:
    """Probability-weighted risk factors across all regimes.

    Returns blended values: e.g. 80% bear / 20% ranging ->
    leverage = 0.8*0.5 + 0.2*0.75 = 0.55.
    """
    blended = {k: 0.0 for k in RISK_SCALING["bull"] if k != "note"}
    for label, prob in probabilities.items():
        key = _regime_key(label)
        for k in blended:
            blended[k] += prob * RISK_SCALING[key][k]
    dominant = max(probabilities, key=probabilities.get) if probabilities else REGIME_LABELS["ranging"]
    return {
        **{k: round(float(v), 4) for k, v in blended.items()},
        "dominant_regime": dominant,
        "dominant_key": _regime_key(dominant),
        "note": RISK_SCALING[_regime_key(dominant)]["note"],
    }


def _regime_key(label: str) -> str:
    for key, lbl in REGIME_LABELS.items():
        if lbl == label:
            return key
    return "ranging"


def detect_regime(ohlc: pd.DataFrame, window: int = WINDOW, seed: int = 42, use_cache: bool = True) -> dict:
    """Detect the current market regime from daily OHLCV data.

    Args:
        ohlc: DataFrame with open/high/low/close/volume indexed by date.
        window: Number of trailing bars to fit on (default 504, ~2y).
        seed: RNG seed for reproducible GMM fits.
        use_cache: Cache the fit for 5 minutes (endpoint-friendly).

    Returns:
        Dict with regime label, probabilities, state characteristics,
        risk scaling, and the fitted model bundle.
    """
    features = _features(ohlc, window=window)

    cache_key = None
    if use_cache:
        last_ts = ohlc.index[-1]
        cache_key = f"{len(features)}:{last_ts}"
        hit = _cache.get(cache_key)
        if hit and time.time() - hit[0] < _CACHE_TTL:
            model, probs, risk = hit[1]
            return {
                "regime": risk["dominant_regime"],
                "regime_key": risk["dominant_key"],
                "probabilities": probs,
                "risk": risk,
                "model": model,
                "cached": True,
            }

    model = fit_regime_model(ohlc, window=window, seed=seed)
    probs = _probabilities(model, features)
    risk = get_risk_scaling(probs)

    if cache_key and use_cache:
        _cache[cache_key] = (time.time(), (model, probs, risk))

    gmm = model["gmm"]
    mapping = model["state_map"]
    means = gmm.means_
    char = {}
    for i in range(gmm.n_components):
        key = mapping[i]
        char[key] = {
            "mean_log_return": round(float(means[i, 0] * model["scaler"].scale_[0]), 6),
            "volatility_rank": i,
        }

    return {
        "regime": risk["dominant_regime"],
        "regime_key": risk["dominant_key"],
        "probabilities": probs,
        "risk": risk,
        "characteristics": char,
        "model": model,
        "n_samples": len(features),
        "cached": False,
    }


def state_sequence(ohlc: pd.DataFrame, window: int = WINDOW, seed: int = 42) -> pd.Series:
    """Per-day regime key series over the trailing window.

    Useful for regime-conditional rolling performance: weight each model by
    its accuracy on days whose regime matches the current one.
    """
    model = fit_regime_model(ohlc, window=window, seed=seed)
    features = model["features"]
    states = model["gmm"].predict(model["X_scaled"])
    mapping = model["state_map"]
    return pd.Series(
        [mapping[int(s)] for s in states],
        index=features.index,
        name="regime",
    )

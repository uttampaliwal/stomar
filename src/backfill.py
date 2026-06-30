"""Historical backfill: reconstruct past decisions for meta-controller training.

Fetches historical OHLCV data, runs available signal modules at each point
in time, logs decisions + actual outcomes to the ledger. Unavailable signals
(sentiment, flow, PCR, MTF) are filled with neutral defaults.

This gives the meta-controller ~250 training samples immediately,
instead of waiting 2-3 months for live data to accumulate.

Usage:
    from src.backfill import HistoricalBackfill
    from src.ledger import Ledger

    ledger = Ledger()  # uses data/stomar.db by default
    backfill = HistoricalBackfill(ledger)
    summary = backfill.run(tickers=["RELIANCE.NS", "TCS.NS"], lookback_days=252)
"""

import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NEUTRAL_SIGNALS = {
    "sentiment_score": 0.0,
    "fii_net": 0.0,
    "dii_net": 0.0,
    "pcr": 1.0,
    "max_pain": None,
    "mtf_signal": 0.0,
}


class HistoricalBackfill:
    """Reconstruct historical decisions for meta-controller training."""

    def __init__(self, ledger):
        self.ledger = ledger

    def run(self, tickers: list[str], lookback_days: int = 252) -> dict:
        """Backfill historical decisions for all tickers.

        Args:
            tickers: List of ticker symbols
            lookback_days: How many trading days of history to fetch

        Returns:
            Summary with per-ticker results
        """
        summary = {"tickers": {}, "total_decisions": 0, "total_outcomes": 0}

        for ticker in tickers:
            try:
                result = self._backfill_ticker(ticker, lookback_days)
                summary["tickers"][ticker] = result
                summary["total_decisions"] += result["decisions"]
                summary["total_outcomes"] += result["outcomes"]
            except Exception as e:
                logger.error(f"Backfill failed for {ticker}: {e}")
                summary["tickers"][ticker] = {"error": str(e), "decisions": 0, "outcomes": 0}

        logger.info(
            f"Backfill complete: {summary['total_decisions']} decisions, "
            f"{summary['total_outcomes']} outcomes across {len(tickers)} tickers"
        )
        return summary

    def _backfill_ticker(self, ticker: str, lookback_days: int) -> dict:
        """Backfill one ticker's historical data."""
        logger.info(f"Backfilling {ticker} ({lookback_days} days)...")

        df = self._fetch_data(ticker, lookback_days)
        if df is None or len(df) < 60:
            logger.warning(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
            return {"decisions": 0, "outcomes": 0}

        close = df["close"]
        returns = close.pct_change().dropna()

        # Pre-compute regime (uses full history for detection)
        regime_result = None
        try:
            from src.regime import detect_regime
            regime_result = detect_regime(close, ohlc=df)
        except Exception:
            pass

        # Pre-compute ensemble models if available
        models = None
        try:
            from src.model import models_exist, load_models
            if models_exist(ticker):
                tup = load_models(ticker)
                # load_models returns (lstm, gru, transformer, xgb, scaler, features, lgb_model)
                models = {
                    "lstm": tup[0], "gru": tup[1], "transformer": tup[2],
                    "xgb": tup[3], "scaler": tup[4], "features": tup[5], "lgb": tup[6],
                }
        except Exception:
            pass

        decisions = 0
        outcomes = 0
        warmup = 60  # Need 60 days of history for features

        for i in range(warmup, len(df) - 1):
            date = str(df.index[i].date()) if hasattr(df.index[i], 'date') else str(df.index[i])[:10]
            current_close = close.iloc[i]

            # Compute regime using only data up to current day (no look-ahead)
            day_regime_result = None
            try:
                from src.regime import detect_regime
                day_close = close.iloc[:i + 1]
                day_df = df.iloc[:i + 1]
                day_regime_result = detect_regime(day_close, ohlc=day_df)
            except Exception:
                pass

            # Compute signals available from historical data
            signals = self._compute_historical_signals(
                df=df, index=i, close=close, returns=returns,
                regime_result=day_regime_result, models=models, ticker=ticker,
            )

            # Make decision using available signals
            decision = self._make_historical_decision(signals)

            # Log decision
            decision_id = self.ledger.log_decision(
                date=date,
                ticker=ticker,
                signals=signals,
                action=decision["action"],
                position_size=decision["position_size"],
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
            )
            decisions += 1

            # Log actual outcome (next day's return)
            next_close = close.iloc[i + 1]
            actual_return = (next_close - current_close) / current_close if current_close != 0 else 0
            actual_direction = 1 if actual_return > 0 else 0

            self.ledger.log_outcome(decision_id, actual_return, actual_direction)
            outcomes += 1

        logger.info(f"{ticker}: {decisions} decisions, {outcomes} outcomes")
        return {"decisions": decisions, "outcomes": outcomes}

    def _fetch_data(self, ticker: str, lookback_days: int) -> pd.DataFrame | None:
        """Fetch historical OHLCV data."""
        try:
            from src.data_fetcher import fetch_stock_data
            return fetch_stock_data(ticker, period="1y")
        except Exception as e:
            logger.warning(f"Data fetch failed for {ticker}: {e}")
            return None

    def _compute_historical_signals(self, df: pd.DataFrame, index: int,
                                    close: pd.Series, returns: pd.Series,
                                    regime_result: dict | None,
                                    models: dict | None,
                                    ticker: str) -> dict:
        """Compute all signals available from historical OHLCV data."""
        signals = {}

        # Current slice up to this point
        df_slice = df.iloc[:index + 1]
        close_slice = close.iloc[:index + 1]
        returns_slice = returns.iloc[:index] if index > 0 else returns.iloc[:1]

        # 1. Regime detection (needed by ensemble weights)
        if regime_result is not None and len(returns_slice) >= 20:
            signals["regime"] = regime_result.get("regime", "Sideways")
            signals["regime_confidence"] = regime_result.get("confidence", 0.5)
        else:
            signals["regime"] = "Sideways"
            signals["regime_confidence"] = 0.5

        # 2. Ensemble prediction (if models exist)
        if models is not None:
            signals.update(self._run_ensemble_historical(models, df_slice, ticker, regime=signals.get("regime")))

        # 3. Sentiment — not available historically, use neutral
        signals["sentiment_score"] = 0.0

        # 4. FII/DII flow — not available historically, use neutral
        signals["fii_net"] = 0.0
        signals["dii_net"] = 0.0

        # 5. Options PCR — not available historically, use neutral
        signals["pcr"] = 1.0
        signals["max_pain"] = None

        # 6. Multi-timeframe — partially reconstructable
        signals["mtf_signal"] = self._compute_simple_mtf(close_slice)

        # 7. Risk metrics
        signals.update(self._compute_risk_historical(returns_slice))

        # 8. Volatility forecast
        signals.update(self._compute_volatility_historical(returns_slice))

        # 9. Fundamentals — use current (changes slowly, acceptable for backfill)
        signals.update(self._compute_fundamentals_historical(ticker))

        return signals

    def _run_ensemble_historical(self, models: dict, df_slice: pd.DataFrame,
                                 ticker: str, regime: str = None) -> dict:
        """Run ensemble on historical slice."""
        try:
            from src.features import add_technical_indicators
            from src.ensemble import predict_ensemble

            feature_cols = models["features"]
            df_feat = add_technical_indicators(df_slice)

            # Fill missing model features with 0 (signal features not available in historical OHLCV)
            for col in feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0

            existing_feats = [c for c in feature_cols if c in df_feat.columns]
            df_feat = df_feat.dropna(subset=existing_feats)

            if len(df_feat) < 2:
                return {}

            # predict_ensemble returns (direction, confidence, details)
            direction, confidence, _ = predict_ensemble(
                models["lstm"], models["gru"], models["transformer"],
                models["xgb"], models["scaler"], feature_cols, df_feat,
                lgb_model=models.get("lgb"), regime=regime,
            )
            return {
                "ensemble_direction": direction,
                "ensemble_confidence": confidence / 100.0 if confidence else None,
            }
        except Exception as e:
            logger.debug(f"Ensemble failed for {ticker}: {e}")
            return {}

    def _compute_simple_mtf(self, close: pd.Series) -> float:
        """Simple multi-timeframe signal from daily data."""
        if len(close) < 50:
            return 0.0
        sma20 = close.rolling(20).mean().iloc[-1]
        sma50 = close.rolling(50).mean().iloc[-1]
        current = close.iloc[-1]
        if current > sma20 > sma50:
            return 0.8  # Strong uptrend
        elif current > sma20:
            return 0.4  # Mild uptrend
        elif current < sma20 < sma50:
            return -0.8  # Strong downtrend
        elif current < sma20:
            return -0.4  # Mild downtrend
        return 0.0

    def _compute_risk_historical(self, returns: pd.Series) -> dict:
        """Compute risk metrics from historical returns."""
        try:
            from src.risk import calculate_var, calculate_cvar, calculate_sharpe
            if len(returns) < 20:
                return {}
            arr = returns.values.astype(float)
            return {
                "var_95": calculate_var(arr, 0.95),
                "cvar_95": calculate_cvar(arr, 0.95),
                "sharpe": calculate_sharpe(arr),
            }
        except Exception:
            return {}

    def _compute_volatility_historical(self, returns: pd.Series) -> dict:
        """Compute volatility forecast from historical returns."""
        try:
            from src.volatility import forecast_volatility
            if len(returns) < 20:
                return {}
            result = forecast_volatility(returns.values.astype(float))
            return {"volatility_forecast": result.get("forecast", 0) if isinstance(result, dict) else 0}
        except Exception:
            return {}

    def _compute_fundamentals_historical(self, ticker: str) -> dict:
        """Get current fundamentals (changes slowly, acceptable for backfill)."""
        try:
            from src.ranking import fetch_fundamentals, fundamental_score
            fund = fetch_fundamentals(ticker)
            if fund:
                return {"fundamental_score": fundamental_score(fund)}
        except Exception:
            pass
        return {}

    def _make_historical_decision(self, signals: dict) -> dict:
        """Make a decision based on available historical signals."""
        ensemble_dir = signals.get("ensemble_direction")
        ensemble_conf = signals.get("ensemble_confidence", 0) or 0
        sentiment = signals.get("sentiment_score", 0) or 0
        regime = signals.get("regime", "Sideways")

        score = 0.0
        n_signals = 0

        # Ensemble (strongest signal if available)
        if ensemble_dir is not None:
            if ensemble_dir == 1:
                score += 0.4 * min(ensemble_conf * 2, 1.0)
            else:
                score -= 0.4 * min(ensemble_conf * 2, 1.0)
            n_signals += 1

        # Sentiment (neutral in backfill)
        if sentiment != 0:
            score += 0.2 * np.sign(sentiment)
            n_signals += 1

        # Regime
        if regime == "Bull":
            score += 0.1
        elif regime == "Bear":
            score -= 0.1
        n_signals += 1

        # MTF
        mtf = signals.get("mtf_signal", 0) or 0
        if mtf != 0:
            score += 0.15 * np.sign(mtf)
            n_signals += 1

        # Risk: high VaR → reduce exposure
        var_95 = signals.get("var_95", 0) or 0
        if var_95 < -0.03:
            score -= 0.1
            n_signals += 1

        # Convert to action
        if n_signals == 0:
            action = "HOLD"
            size = 0.0
        elif score > 0.15:
            action = "BUY"
            size = min(0.05, abs(score) * 0.15)
        elif score < -0.15:
            action = "SELL"
            size = min(0.05, abs(score) * 0.15)
        else:
            action = "HOLD"
            size = 0.0

        return {
            "action": action,
            "position_size": round(size, 4),
            "confidence": round(min(1.0, abs(score)), 4),
            "reasoning": f"Backfill: score={score:.3f}, regime={regime}",
        }

"""Daily orchestrator: runs the full signal pipeline once per day.

Pulls fresh data, runs all 14 signal modules, passes outputs to
the meta-controller, and logs everything to the ledger.

Usage:
    from src.signals.orchestrator import DailyOrchestrator
    from src.trading.ledger import Ledger

    ledger = Ledger()  # uses data/stomar.db by default
    orch = DailyOrchestrator(tickers=["RELIANCE.NS", "TCS.NS"], ledger=ledger)
    summary = orch.run()
"""

import logging
import os
from datetime import datetime

import pandas as pd

from src.core.settings import settings

logger = logging.getLogger(__name__)


class DailyOrchestrator:
    """Runs the full signal pipeline once per day."""

    def __init__(self, tickers: list[str], ledger, meta_controller=None,
                 paper_trader=None):
        self.tickers = tickers
        self.ledger = ledger
        self.meta_controller = meta_controller
        self.paper_trader = paper_trader

    def run(self, date: str = None, dry_run: bool = False,
            resolve_outcomes: bool = False) -> dict:
        """Execute one full daily cycle.

        Args:
            date: Trade date (YYYY-MM-DD). Defaults to today.
            dry_run: If True, run signals but don't write to ledger.
            resolve_outcomes: If True, resolve yesterday's unresolved
                decisions with their actual next-day returns (P3.3).

        Returns:
            Summary dict with decisions, trades, errors.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if not self.tickers:
            logger.warning("No tickers configured for daily run")
            return {"date": date, "decisions": [], "trades": [], "errors": []}

        from src.core.trading_mode import get_trading_mode
        mode = get_trading_mode()
        logger.info("orchestrator mode: %s (dry_run=%s)", mode.value, dry_run)

        # Resolve yesterday's decisions before making new ones, so the
        # meta-controller learns from real forward outcomes (not backfill).
        outcome_summary = {"resolved": 0, "failed": 0, "pending": 0}
        if resolve_outcomes and not dry_run:
            outcome_summary = self.resolve_pending_outcomes()
            logger.info(
                "Outcome resolution: %d resolved, %d failed, %d pending",
                outcome_summary["resolved"], outcome_summary["failed"],
                outcome_summary["pending"],
            )

        # Pre-fetch market-wide signals (same for all tickers)
        self._flow_signals = self._run_flow()
        self._pcr_signals = self._run_pcr()

        summary = {"date": date, "decisions": [], "trades": [], "errors": [],
                   "outcomes": outcome_summary}
        trade_count = 0
        total_exposure = 0.0

        for ticker in self.tickers:
            try:
                result = self._process_ticker(ticker, date, dry_run)

                # Enforce daily trade limit
                if result["action"] != "HOLD":
                    if trade_count >= settings.max_daily_trades:
                        logger.info(f"{ticker}: trade blocked (daily limit {settings.max_daily_trades} reached)")
                        result["action"] = "HOLD"
                        result["position_size"] = 0.0
                        result["reasoning"] = f"Daily trade limit ({settings.max_daily_trades}) reached"
                    elif total_exposure + result["position_size"] > settings.max_portfolio_exposure:
                        logger.info(f"{ticker}: trade blocked (portfolio exposure would exceed {settings.max_portfolio_exposure:.0%})")
                        result["action"] = "HOLD"
                        result["position_size"] = 0.0
                        result["reasoning"] = f"Portfolio exposure cap ({settings.max_portfolio_exposure:.0%}) reached"
                    else:
                        trade_count += 1
                        total_exposure += result["position_size"]

                summary["decisions"].append(result)

                # Log the FINAL decision — including blocked trades as
                # HOLD — so the meta-controller never trains on signals
                # for orders that were never placed (#50).
                if not dry_run and result.get("signals"):
                    decision_id = self.ledger.log_decision(
                        date=date,
                        ticker=ticker,
                        signals=result["signals"],
                        action=result["action"],
                        position_size=result["position_size"],
                        confidence=result["confidence"],
                        reasoning=result["reasoning"],
                    )
                    result["decision_id"] = decision_id

                # Execute paper trade if action is BUY or SELL
                if not dry_run and result["action"] in ("BUY", "SELL") and result["position_size"] > 0:
                    trade = self._execute_paper_trade(ticker, result)
                    if trade:
                        summary["trades"].append(trade)

            except Exception as e:
                logger.error(f"Failed to process {ticker}: {e}")
                summary["errors"].append({"ticker": ticker, "error": str(e)})

        logger.info(
            f"Daily run complete: {len(summary['decisions'])} decisions, "
            f"{trade_count} trades, {len(summary['errors'])} errors, "
            f"exposure={total_exposure:.1%}"
        )
        return summary

    def _process_ticker(self, ticker: str, date: str, dry_run: bool) -> dict:
        """Process a single ticker through the full signal pipeline."""
        logger.info(f"Processing {ticker}...")

        signals = self._collect_signals(ticker)
        decision = self._make_decision(signals)

        result = {
            "ticker": ticker,
            "signals": {k: v for k, v in signals.items() if v is not None},
            "action": decision["action"],
            "position_size": decision["position_size"],
            "confidence": decision["confidence"],
            "reasoning": decision["reasoning"],
            "current_price": signals.get("current_price", 0),
        }

        # P4.3: append SHAP top-5 to the decision reasoning
        explanation = signals.get("explanation") or []
        if explanation and decision["reasoning"]:
            parts = []
            for f in explanation[:5]:
                feat = f.get("feature") or "?"
                shap = f.get("shap")
                parts.append(f"{feat}({shap:+.3f})" if shap is not None else feat)
            result["reasoning"] = f"{decision['reasoning']} | SHAP: {', '.join(parts)}"

        logger.info(f"{ticker}: {decision['action']} (size={decision['position_size']:.2%})")
        return result

    def resolve_pending_outcomes(self) -> dict:
        """Resolve unresolved decisions with their actual next-day returns.

        For every live decision without an outcome, find the close on the
        decision date and the close on the next trading day, then log the
        realized return via ``ledger.log_outcome``. Backfill decisions are
        skipped (they are resolved synchronously during backfill).

        Returns:
            {"resolved": int, "failed": int, "pending": int}
        """
        pending = self.ledger.get_unresolved_decisions()
        resolved = 0
        failed = 0
        for d in pending:
            if d.get("source") == "backfill":
                continue
            try:
                result = self._next_day_return(d["ticker"], d["date"])
                if result is None:
                    failed += 1
                    continue
                ret, direction = result
                self.ledger.log_outcome(d["id"], float(ret), int(direction))
                resolved += 1
            except Exception as e:
                logger.debug("Outcome resolution failed for decision %s: %s", d.get("id"), e)
                failed += 1
        return {"resolved": resolved, "failed": failed, "pending": len(pending)}

    def _next_day_return(self, ticker: str, decision_date: str):
        """Return (next_day_return, direction) for a decision date, or None.

        Uses close-to-close returns on the next trading day — the same
        convention the backfill uses, so meta-controller training labels
        stay consistent.
        """
        df = self._fetch_data(ticker)
        if df is None or len(df) < 2:
            return None

        target = pd.Timestamp(decision_date)
        if target not in df.index:
            # Decision date not found in data — resolve at first date after it
            later = df.index[df.index > target]
            if len(later) == 0:
                return None
            entry_idx = df.index.get_loc(later[0])
        else:
            entry_idx = df.index.get_loc(target)

        if entry_idx + 1 >= len(df):
            return None  # no next trading day yet

        entry_close = float(df["close"].iloc[entry_idx])
        next_close = float(df["close"].iloc[entry_idx + 1])
        if entry_close <= 0:
            return None
        ret = (next_close - entry_close) / entry_close
        direction = 1 if ret > 0 else 0
        return ret, direction

    # Stop-loss percentage applied to every new position (settings.stop_loss_pct from entry)

    def _execute_paper_trade(self, ticker: str, result: dict) -> dict | None:
        """Execute a paper trade and log to ledger. Returns trade info or None.

        After every entry order is filled, a protective stop-loss order is
        submitted automatically:
          BUY  → SELL STOP_MARKET at fill_price * (1 - STOP_LOSS_PCT)
          SELL → BUY  STOP_MARKET at fill_price * (1 + STOP_LOSS_PCT)
        """
        try:
            from src.trading.paper_trader import PaperTrader
            from src.trading.engine import OrderSide, OrderType
            from src.data.data_fetcher import get_live_price

            trader = self.paper_trader
            if trader is None:
                from src.core.constants import PAPER_STATE_PATH
                trader = PaperTrader(initial_capital=200000, persist_risk_state=True)
                if os.path.exists(PAPER_STATE_PATH):
                    try:
                        trader.load_state(PAPER_STATE_PATH)
                    except Exception:
                        pass

            price = get_live_price(ticker)
            if price is None or price <= 0:
                logger.warning(f"Cannot get price for {ticker}, skipping trade")
                return None

            equity = trader.get_equity()
            max_position_value = equity * trader.risk_controller.limits.max_position_pct
            position_value = min(equity * result["position_size"], max_position_value)
            risk_result = trader.risk_controller.check_order(
                order_value=position_value,
                current_holdings_value=sum(p.market_value for p in trader.positions.values()),
                ticker=ticker,
                holdings=trader.positions,
            )
            if not risk_result["approved"]:
                logger.info(f"Risk blocked {ticker}: {risk_result.get('reason', 'risk check failed')}")
                return None

            quantity = int(position_value / price) if price > 0 else 0
            if quantity <= 0:
                return None

            side = OrderSide.BUY if result["action"] == "BUY" else OrderSide.SELL
            # Use execute_market_trade for instant fill at live price
            order = trader.execute_market_trade(ticker, side, quantity, price=price)

            if order is None or order.status.name != "FILLED":
                logger.warning(f"Order not filled for {ticker}: {order}")
                return None

            fill_price = order.filled_price if order.filled_price > 0 else price

            # ── Attach protective stop-loss ──────────────────────────────────
            if side == OrderSide.BUY:
                stop_price = round(fill_price * (1 - settings.stop_loss_pct), 2)
                stop_side = OrderSide.SELL
            else:
                stop_price = round(fill_price * (1 + settings.stop_loss_pct), 2)
                stop_side = OrderSide.BUY

            stop_order = trader.place_order(
                ticker, stop_side, OrderType.STOP_MARKET,
                quantity, stop_price=stop_price,
            )
            if stop_order is not None:
                logger.info(
                    f"Stop-loss placed: {stop_side.value} {quantity} {ticker} "
                    f"@ stop ₹{stop_price:.2f} (entry ₹{fill_price:.2f}, "
                    f"{settings.stop_loss_pct:.0%} risk)"
                )
            # ────────────────────────────────────────────────────────────────

            # Log entry trade to ledger
            decision_id = result.get("decision_id")
            if decision_id:
                self.ledger.log_trade(
                    decision_id=decision_id,
                    ticker=ticker,
                    side=result["action"],
                    quantity=quantity,
                    price=fill_price,
                )

            trade_info = {
                "ticker": ticker,
                "side": result["action"],
                "quantity": quantity,
                "price": round(fill_price, 2),
                "position_size": result["position_size"],
                "stop_price": stop_price,
            }
            logger.info(
                f"Executed paper trade: {result['action']} {quantity} {ticker} "
                f"@ ₹{fill_price:.2f} | stop ₹{stop_price:.2f}"
            )
            return trade_info

        except Exception as e:
            logger.error(f"Paper trade execution failed for {ticker}: {e}")
            return None

    def _collect_signals(self, ticker: str) -> dict:
        """Run all signal modules and collect their outputs."""
        signals = {}

        # 1. Fetch price data
        df = self._fetch_data(ticker)
        if df is None or len(df) < 50:
            logger.warning(f"{ticker}: insufficient data ({len(df) if df is not None else 0} rows)")
            return signals

        close = df["close"]
        returns = close.pct_change().dropna()
        signals["current_price"] = float(close.iloc[-1])

        # 2. Collect auxiliary signals FIRST (needed as ensemble features)
        signals.update(self._run_sentiment(ticker))
        signals.update(getattr(self, "_flow_signals", {}))
        signals.update(getattr(self, "_pcr_signals", {}))
        signals.update(self._run_mtf(ticker, df))
        signals.update(self._run_regime(close, df))
        signals.update(self._run_risk(returns))
        signals.update(self._run_volatility(returns))
        signals.update(self._run_fundamentals(ticker))

        # 3. Ensemble prediction (uses auxiliary signals as features)
        signals.update(self._run_ensemble(ticker, df, signals))

        return signals

    def _fetch_data(self, ticker: str) -> pd.DataFrame | None:
        """Fetch latest OHLCV data with validation + auto-repair (P2.1/P2.2)."""
        try:
            from src.data.data_fetcher import fetch_validated_stock_data
            result = fetch_validated_stock_data(ticker, period="1y")
            return result["df"]
        except Exception as e:
            logger.warning(f"Data fetch failed for {ticker}: {e}")
            return None

    def _run_ensemble(self, ticker: str, df: pd.DataFrame, signals: dict = None) -> dict:
        """Run ensemble prediction, using a per-ticker meta-learner when available."""
        try:
            from src.models.trainer_features import build_feature_frame
            from src.models.model import load_models, models_exist, model_feature_cols
            from src.models.ensemble import predict_ensemble, load_meta_model
            from src.core.constants import MODELS_DIR
            import os

            if not models_exist(ticker):
                return {}

            tup = load_models(ticker)
            models = {
                "lstm": tup[0], "gru": tup[1], "transformer": tup[2],
                "xgb": tup[3], "scaler": tup[4], "features": tup[5], "lgb": tup[6],
            }
            feature_cols = model_feature_cols(models)  # model's own trained schema

            # Load per-ticker meta-learner if available (trained in trainer.py)
            meta_model = None
            ticker_meta_path = os.path.join(
                MODELS_DIR, f"meta_{ticker.replace('.', '_')}.pkl"
            )
            if os.path.exists(ticker_meta_path):
                try:
                    meta_model = load_meta_model(ticker_meta_path)
                    logger.debug(f"Loaded per-ticker meta-learner for {ticker}")
                except Exception as e:
                    logger.debug(f"Could not load per-ticker meta for {ticker}: {e}")

            # Trainer-identical feature frame (O2 parity): the full SOTA factor
            # set, not the bare legacy indicators. Freshly computed signals
            # below override the cache-derived PIT values on the last row.
            df_feat = build_feature_frame(df, ticker=ticker)

            # Inject real-time signal values as features for the model
            signal_map = {
                "sentiment_score": "sentiment_score",
                "fii_net": "fii_net",
                "dii_net": "dii_net",
                "pcr": "pcr",
                "mtf_signal": "mtf_signal",
            }
            for feat_name, sig_name in signal_map.items():
                if feat_name in feature_cols and feat_name not in df_feat.columns:
                    val = (signals or {}).get(sig_name, 0) or 0
                    df_feat[feat_name] = val

            # Fill any remaining missing model features with 0
            for col in feature_cols:
                if col not in df_feat.columns:
                    df_feat[col] = 0

            existing_feats = [c for c in feature_cols if c in df_feat.columns]
            df_feat = df_feat.dropna(subset=existing_feats)

            if len(df_feat) < 2:
                return {}

            regime = signals.get("regime") if signals else None

            ensemble_dir, confidence, details = predict_ensemble(
                models["lstm"], models["gru"], models["transformer"],
                models["xgb"], models["scaler"], feature_cols, df_feat,
                lgb_model=models["lgb"],
                meta_model=meta_model,
                regime=regime,
            )
            result = {
                "ensemble_direction": ensemble_dir,
                "ensemble_confidence": confidence / 100.0 if confidence else None,
            }
            # P4.3: SHAP top-5 explanation for the production decision
            try:
                from src.signals.interpretability import explain_prediction
                explanation = explain_prediction(ticker, df_feat, feature_cols, top_n=5)
                top = explanation.get("top") or []
                result["explanation"] = [
                    {"feature": f.get("feature"), "shap": f.get("shap_value"),
                     "value": f.get("current_value")}
                    for f in top[:5]
                    if f.get("feature")
                ]
            except Exception as e:
                logger.debug(f"Explanation failed for {ticker}: {e}")
            return result
        except Exception as e:
            logger.debug(f"Ensemble failed for {ticker}: {e}")
            return {}

    def _run_sentiment(self, ticker: str) -> dict:
        """Run sentiment analysis."""
        try:
            from src.signals.sentiment import get_stock_sentiment
            result = get_stock_sentiment(ticker)
            return {"sentiment_score": result.get("score", 0)}
        except Exception as e:
            logger.debug(f"Sentiment failed for {ticker}: {e}")
            return {}

    def _run_flow(self) -> dict:
        """Run FII/DII flow analysis."""
        try:
            from src.signals.flow import fetch_fii_dii
            fii_dii = fetch_fii_dii()
            if fii_dii is not None and len(fii_dii) > 0:
                last = fii_dii.iloc[-1]
                return {
                    "fii_net": float(last.get("fii_net", 0) or 0),
                    "dii_net": float(last.get("dii_net", 0) or 0),
                }
        except Exception as e:
            logger.debug(f"Flow failed: {e}")
        return {}

    def _run_pcr(self) -> dict:
        """Run options PCR analysis."""
        try:
            from src.signals.flow import fetch_options_pcr
            pcr_data = fetch_options_pcr()
            if pcr_data:
                return {
                    "pcr": pcr_data.get("pcr"),
                    "max_pain": pcr_data.get("max_pain"),
                }
        except Exception as e:
            logger.debug(f"PCR failed: {e}")
        return {}

    def _run_mtf(self, ticker: str, df: pd.DataFrame) -> dict:
        """Run multi-timeframe analysis."""
        try:
            from src.signals.multitimeframe import fetch_mtf_data, get_combined_signal
            mtf_data = fetch_mtf_data(ticker)
            if not mtf_data:
                return {}
            signal = get_combined_signal(mtf_data)
            return {
                "mtf_signal": signal.get("direction", 0) if isinstance(signal, dict) else 0,
                "mtf_confidence": signal.get("confidence", 0) if isinstance(signal, dict) else 0,
            }
        except Exception as e:
            logger.debug(f"MTF failed for {ticker}: {e}")
            return {}

    def _run_regime(self, close: pd.Series, df: pd.DataFrame = None) -> dict:
        """Run regime detection."""
        try:
            from src.signals.regime import detect_regime
            if len(close) < 20:
                return {}
            result = detect_regime(close, ohlc=df)
            return {
                "regime": result.get("regime", "Sideways"),
                "regime_confidence": result.get("confidence", 0.5),
            }
        except Exception as e:
            logger.debug(f"Regime failed: {e}")
            return {}

    def _run_risk(self, returns: pd.Series) -> dict:
        """Run risk metrics."""
        try:
            from src.trading.risk import calculate_var, calculate_cvar, calculate_sharpe
            if len(returns) < 20:
                return {}
            returns_arr = returns.values.astype(float)
            return {
                "var_95": calculate_var(returns_arr, 0.95),
                "cvar_95": calculate_cvar(returns_arr, 0.95),
                "sharpe": calculate_sharpe(returns_arr),
            }
        except Exception as e:
            logger.debug(f"Risk failed: {e}")
            return {}

    def _run_volatility(self, returns: pd.Series) -> dict:
        """Run volatility forecast."""
        try:
            from src.signals.volatility import forecast_volatility
            if len(returns) < 20:
                return {}
            result = forecast_volatility(returns.values.astype(float))
            return {"volatility_forecast": result.get("current_vol", 0) if isinstance(result, dict) else 0}
        except Exception as e:
            logger.debug(f"Volatility failed: {e}")
            return {}

    def _run_fundamentals(self, ticker: str) -> dict:
        """Run fundamental analysis."""
        try:
            from src.signals.ranking import fetch_fundamentals, fundamental_score
            fund = fetch_fundamentals(ticker)
            if fund:
                score = fundamental_score(fund)
                return {"fundamental_score": score}
        except Exception as e:
            logger.debug(f"Fundamentals failed for {ticker}: {e}")
        return {}

    def _make_decision(self, signals: dict) -> dict:
        """Pass signals to meta-controller for a decision."""
        if not signals:
            return {
                "action": "HOLD",
                "position_size": 0.0,
                "confidence": 0.0,
                "reasoning": "No signals collected",
            }
        if self.meta_controller is not None:
            return self.meta_controller.decide(signals)
        return self._default_decision(signals)

    def _default_decision(self, signals: dict) -> dict:
        """Simple rule-based fallback when no meta-controller is available."""
        from src.core.settings import settings
        ensemble_dir = signals.get("ensemble_direction")
        ensemble_conf = signals.get("ensemble_confidence", 0) or 0
        buy_threshold = getattr(settings, "buy_threshold", 0.6)
        sell_threshold = getattr(settings, "sell_threshold", 0.4)
        max_pos = getattr(settings, "max_position_pct", 0.10)

        if ensemble_dir == 1 and ensemble_conf > buy_threshold:
            action = "BUY"
            size = min(max_pos, (ensemble_conf - 0.5) * 0.5)
        elif ensemble_dir == 0 and ensemble_conf > sell_threshold:
            action = "SELL"
            size = min(max_pos, (ensemble_conf - 0.5) * 0.5)
        else:
            action = "HOLD"
            size = 0.0

        return {
            "action": action,
            "position_size": round(size, 4),
            "confidence": round(ensemble_conf, 4),
            "reasoning": f"Default: ensemble_dir={ensemble_dir}, conf={ensemble_conf:.2f}",
        }

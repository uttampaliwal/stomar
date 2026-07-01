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
from datetime import datetime

import pandas as pd

logger = logging.getLogger(__name__)


class DailyOrchestrator:
    """Runs the full signal pipeline once per day."""

    MAX_DAILY_TRADES = 5
    MAX_PORTFOLIO_EXPOSURE = 0.50

    def __init__(self, tickers: list[str], ledger, meta_controller=None,
                 paper_trader=None):
        self.tickers = tickers
        self.ledger = ledger
        self.meta_controller = meta_controller
        self.paper_trader = paper_trader

    def run(self, date: str = None, dry_run: bool = False) -> dict:
        """Execute one full daily cycle.

        Args:
            date: Trade date (YYYY-MM-DD). Defaults to today.
            dry_run: If True, run signals but don't write to ledger.

        Returns:
            Summary dict with decisions, trades, errors.
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")

        if not self.tickers:
            logger.warning("No tickers configured for daily run")
            return {"date": date, "decisions": [], "trades": [], "errors": []}

        summary = {"date": date, "decisions": [], "trades": [], "errors": []}
        trade_count = 0
        total_exposure = 0.0

        for ticker in self.tickers:
            try:
                result = self._process_ticker(ticker, date, dry_run)

                # Enforce daily trade limit
                if result["action"] != "HOLD":
                    if trade_count >= self.MAX_DAILY_TRADES:
                        logger.info(f"{ticker}: trade blocked (daily limit {self.MAX_DAILY_TRADES} reached)")
                        result["action"] = "HOLD"
                        result["position_size"] = 0.0
                        result["reasoning"] = f"Daily trade limit ({self.MAX_DAILY_TRADES}) reached"
                    elif total_exposure + result["position_size"] > self.MAX_PORTFOLIO_EXPOSURE:
                        logger.info(f"{ticker}: trade blocked (portfolio exposure would exceed {self.MAX_PORTFOLIO_EXPOSURE:.0%})")
                        result["action"] = "HOLD"
                        result["position_size"] = 0.0
                        result["reasoning"] = f"Portfolio exposure cap ({self.MAX_PORTFOLIO_EXPOSURE:.0%}) reached"
                    else:
                        trade_count += 1
                        total_exposure += result["position_size"]

                summary["decisions"].append(result)

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

        if not dry_run:
            decision_id = self.ledger.log_decision(
                date=date,
                ticker=ticker,
                signals=signals,
                action=decision["action"],
                position_size=decision["position_size"],
                confidence=decision["confidence"],
                reasoning=decision["reasoning"],
            )
            result["decision_id"] = decision_id

        logger.info(f"{ticker}: {decision['action']} (size={decision['position_size']:.2%})")
        return result

    # Stop-loss percentage applied to every new position (5% from entry)
    STOP_LOSS_PCT: float = 0.05

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
                trader = PaperTrader(initial_capital=200000)

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
            order = trader.place_order(ticker, side, OrderType.MARKET, quantity, price=price)

            if order is None or order.status.value != "FILLED":
                logger.warning(f"Order not filled for {ticker}: {order}")
                return None

            fill_price = order.filled_price if order.filled_price > 0 else price

            # ── Attach protective stop-loss ──────────────────────────────────
            if side == OrderSide.BUY:
                stop_price = round(fill_price * (1 - self.STOP_LOSS_PCT), 2)
                stop_side = OrderSide.SELL
            else:
                stop_price = round(fill_price * (1 + self.STOP_LOSS_PCT), 2)
                stop_side = OrderSide.BUY

            stop_order = trader.place_order(
                ticker, stop_side, OrderType.STOP_MARKET,
                quantity, stop_price=stop_price,
            )
            if stop_order is not None:
                logger.info(
                    f"Stop-loss placed: {stop_side.value} {quantity} {ticker} "
                    f"@ stop ₹{stop_price:.2f} (entry ₹{fill_price:.2f}, "
                    f"{self.STOP_LOSS_PCT:.0%} risk)"
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
        signals.update(self._run_flow())
        signals.update(self._run_pcr())
        signals.update(self._run_mtf(ticker, df))
        signals.update(self._run_regime(close, df))
        signals.update(self._run_risk(returns))
        signals.update(self._run_volatility(returns))
        signals.update(self._run_fundamentals(ticker))

        # 3. Ensemble prediction (uses auxiliary signals as features)
        signals.update(self._run_ensemble(ticker, df, signals))

        return signals

    def _fetch_data(self, ticker: str) -> pd.DataFrame | None:
        """Fetch latest OHLCV data."""
        try:
            from src.data.data_fetcher import fetch_stock_data
            return fetch_stock_data(ticker, period="1y")
        except Exception as e:
            logger.warning(f"Data fetch failed for {ticker}: {e}")
            return None

    def _run_ensemble(self, ticker: str, df: pd.DataFrame, signals: dict = None) -> dict:
        """Run ensemble prediction, using a per-ticker meta-learner when available."""
        try:
            from src.data.features import add_technical_indicators
            from src.models.model import load_models, models_exist
            from src.models.ensemble import predict_ensemble, load_meta_model
            from src.models.trainer import FEATURE_COLS
            from src.core.constants import MODELS_DIR
            import os

            if not models_exist(ticker):
                return {}

            tup = load_models(ticker)
            models = {
                "lstm": tup[0], "gru": tup[1], "transformer": tup[2],
                "xgb": tup[3], "scaler": tup[4], "features": tup[5], "lgb": tup[6],
            }

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

            df_feat = add_technical_indicators(df)

            # Inject real-time signal values as features for the model
            signal_map = {
                "sentiment_score": "sentiment_score",
                "fii_net": "fii_net",
                "dii_net": "dii_net",
                "pcr": "pcr",
                "mtf_signal": "mtf_signal",
            }
            for feat_name, sig_name in signal_map.items():
                if feat_name in FEATURE_COLS and feat_name not in df_feat.columns:
                    val = (signals or {}).get(sig_name, 0) or 0
                    df_feat[feat_name] = val

            # Fill any remaining missing model features with 0
            for col in FEATURE_COLS:
                if col not in df_feat.columns:
                    df_feat[col] = 0

            existing_feats = [c for c in FEATURE_COLS if c in df_feat.columns]
            df_feat = df_feat.dropna(subset=[c for c in existing_feats if c in df_feat.columns] + ["target"])

            if len(df_feat) < 2:
                return {}

            regime = signals.get("regime") if signals else None

            ensemble_dir, confidence, details = predict_ensemble(
                models["lstm"], models["gru"], models["transformer"],
                models["xgb"], models["scaler"], FEATURE_COLS, df_feat,
                lgb_model=models["lgb"],
                meta_model=meta_model,
                regime=regime,
            )
            return {
                "ensemble_direction": ensemble_dir,
                "ensemble_confidence": confidence / 100.0 if confidence else None,
            }
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
        ensemble_dir = signals.get("ensemble_direction")
        ensemble_conf = signals.get("ensemble_confidence", 0) or 0

        if ensemble_dir == 1 and ensemble_conf > 0.55:
            action = "BUY"
            size = min(0.05, (ensemble_conf - 0.5) * 0.5)
        elif ensemble_dir == 0 and ensemble_conf > 0.55:
            action = "SELL"
            size = min(0.05, (ensemble_conf - 0.5) * 0.5)
        else:
            action = "HOLD"
            size = 0.0

        return {
            "action": action,
            "position_size": round(size, 4),
            "confidence": round(ensemble_conf, 4),
            "reasoning": f"Default: ensemble_dir={ensemble_dir}, conf={ensemble_conf:.2f}",
        }

"""Daily autonomous loop. Run via scheduler or manually.

Usage:
    python run_daily.py                          # Run for all NSE stocks
    python run_daily.py --ticker RELIANCE.NS     # Specific tickers
    python run_daily.py --dry-run                # Don't write to ledger
    python run_daily.py --backfill               # Backfill 1 year history + train meta
    python run_daily.py --backfill --days 126    # Backfill 6 months + train
    python run_daily.py --train-meta             # Train meta-controller only
    python run_daily.py --paper-trade            # Auto-execute paper trades
    python run_daily.py --paper-trade --capital 500000
"""

import argparse
import logging
import signal
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))

from src.data.data_fetcher import NSE_STOCKS
from src.trading.ledger import Ledger
from src.signals.orchestrator import DailyOrchestrator
from src.models.meta_controller import MetaController
from src.core.constants import META_CONTROLLER_PATH
from src.core.trading_mode import get_trading_mode, mode_banner, require_live_allowed


def run_paper_trades(decisions: list, ledger, capital: float = 200_000,
                     state_path: str = None):
    """Auto-execute paper trades based on orchestrator decisions."""
    from src.brokers import get_broker
    from src.trading.execution_manager import ExecutionManager
    from src.trading.risk_controls import RiskController
    from src.trading.paper_trader import PaperTrader
    from src.trading.engine import OrderSide, OrderType

    broker = get_broker()  # dry-run unless live gate fully passed (not used here)
    risk = RiskController()
    manager = ExecutionManager(broker, risk)

    trader = PaperTrader(initial_capital=capital)
    trader.load_state(state_path)

    print("\n=== Paper Trading ===")
    print(f"Capital:    Rs. {trader.initial_capital:,.0f}")
    print(f"Cash:       Rs. {trader.cash:,.0f}")
    print(f"Equity:     Rs. {trader.get_equity():,.0f}")

    executed = 0
    for d in decisions:
        ticker = d["ticker"]
        action = d["action"]
        conf = d.get("confidence", 0)
        size_pct = d.get("position_size", 0)

        if action == "HOLD" or size_pct <= 0:
            continue

        price = d.get("current_price", 0)
        if price <= 0:
            continue

        equity = trader.get_equity()
        invest_amount = equity * size_pct
        qty = max(1, int(invest_amount / price))

        gate = manager.gate_order(ticker, action, qty,
                                  order_value=invest_amount)
        if not gate["approved"]:
            print(f"  BLOCK {ticker:15s} gate: {gate['reason']}")
            continue

        if action == "BUY":
            from src.data.data_fetcher import get_live_price
            current_price = get_live_price(ticker)
            if not current_price or current_price <= 0:
                current_price = price

            order = trader.place_order(
                ticker, OrderSide.BUY, OrderType.MARKET, qty, price=current_price,
            )
            filled = trader.on_bar(ticker, current_price, current_price, current_price, current_price)
            if filled and any(r.side == "BUY" for r in filled):
                executed += 1
                print(f"  BUY  {qty:4d} {ticker:15s} @ Rs.{current_price:.2f}  (conf={conf:.2f})")

        elif action == "SELL":
            sell_qty = qty
            if ticker in trader.positions and trader.positions[ticker].quantity > 0:
                sell_qty = min(qty, trader.positions[ticker].quantity)
            from src.data.data_fetcher import get_live_price
            current_price = get_live_price(ticker)
            if not current_price or current_price <= 0:
                current_price = price

            order = trader.place_order(
                ticker, OrderSide.SELL, OrderType.MARKET, sell_qty, price=current_price,
            )
            filled = trader.on_bar(ticker, current_price, current_price, current_price, current_price)
            if filled and any(r.side == "SELL" for r in filled):
                executed += 1
                print(f"  SELL {sell_qty:4d} {ticker:15s} @ Rs.{current_price:.2f}  (conf={conf:.2f})")

    # Update current prices for open positions
    for ticker in list(trader.positions.keys()):
        try:
            from src.data.data_fetcher import get_live_price
            current_price = get_live_price(ticker)
            if current_price and current_price > 0:
                trader.update_prices({ticker: current_price})
        except Exception:
            pass

    summary = trader.get_summary()
    print(f"\nExecuted: {executed} trades")
    print(f"Equity:   Rs. {summary['current_equity']:,.0f}")
    print(f"Return:   {summary['total_return_pct']:.2%}")
    print(f"Trades:   {summary['total_trades']}")
    if summary['open_positions']:
        print("Open positions:")
        for t, p in summary['open_positions'].items():
            print(f"  {t}: {p['quantity']} @ Rs.{p['avg_cost']:.2f} (P&L: Rs.{p['unrealized_pnl']:,.0f})")

    trader.save_state(state_path)
    return summary


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    shutdown_requested = False

    def _handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name
        logging.warning("Received %s — shutting down gracefully", sig_name)
        shutdown_requested = True

    signal.signal(signal.SIGTERM, _handle_shutdown)
    signal.signal(signal.SIGINT, _handle_shutdown)

    parser = argparse.ArgumentParser(description="StoMar Daily Signal Loop")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="Tickers to process (default: all NSE stocks)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run signals but don't write to ledger")
    parser.add_argument("--db", default=None,
                        help="SQLite ledger path (default: data/stomar.db)")
    parser.add_argument("--backfill", action="store_true",
                        help="Backfill historical data, then train meta-controller")
    parser.add_argument("--days", type=int, default=252,
                        help="Trading days to backfill (default: 252 = ~1 year)")
    parser.add_argument("--train-meta", action="store_true",
                        help="Train meta-controller on ledger history")
    parser.add_argument("--paper-trade", action="store_true",
                        help="Auto-execute paper trades based on signals")
    parser.add_argument("--capital", type=float, default=200_000,
                        help="Paper trading capital (default: 200000)")
    parser.add_argument("--schedule-hours", type=float, default=0,
                        help="Run the loop repeatedly every N hours (0 = disabled)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Run data retention cleanup after daily loop (or standalone)")
    parser.add_argument("--cleanup-dry-run", action="store_true",
                        help="Dry-run mode for cleanup: report what would be deleted")
    args = parser.parse_args()

    tickers = args.ticker if args.ticker else NSE_STOCKS
    ledger = Ledger(args.db)  # None uses default data/stomar.db

    # --- Cleanup-only mode ---
    if args.cleanup or args.cleanup_dry_run:
        from src.data.data_retention import DataRetentionPolicy, print_report
        policy = DataRetentionPolicy()
        report = policy.run_all(
            active_tickers=tickers,
            dry_run=args.cleanup_dry_run,
        )
        print_report(report)
        ledger.close()
        sys.exit(0)

    # --- Backfill mode ---
    if args.backfill:
        print("=== StoMar Historical Backfill ===")
        print(f"Tickers:  {len(tickers)}")
        print(f"Days:     {args.days}")
        print(f"Ledger:   {args.db}")
        print()

        from src.core.backfill import HistoricalBackfill
        backfill = HistoricalBackfill(ledger)
        bf_summary = backfill.run(tickers=tickers, lookback_days=args.days)

        print("\n=== Backfill Summary ===")
        print(f"Total decisions: {bf_summary['total_decisions']}")
        print(f"Total outcomes:  {bf_summary['total_outcomes']}")
        for ticker, result in bf_summary["tickers"].items():
            if "error" in result:
                print(f"  [ERR]  {ticker}: {result['error']}")
            else:
                print(f"  [OK]   {ticker}: {result['decisions']} decisions, {result['outcomes']} outcomes")

        # Auto-train meta-controller after backfill
        print("\n=== Training Meta-Controller ===")
        mc = MetaController()
        result = mc.train(ledger)
        print(f"Status: {result['status']}")
        if result["status"] == "trained":
            print(f"Accuracy: {result['accuracy']:.1%}")
            print(f"Samples:  {result['n_samples']}")
            weights = mc.get_weights()
            print("Top signals:")
            for name, weight in list(weights.items())[:5]:
                direction = "positive" if weight > 0 else "negative"
                print(f"  {name}: {weight:+.4f} ({direction})")

            # Save trained model (hash-verified artifact bundle)
            mc.save()
            print(f"\nMeta-controller saved to {META_CONTROLLER_PATH}")
        else:
            print(f"Not enough data: {result.get('n_samples', 0)} samples (need {result.get('required', 100)})")

        ledger.close()
        sys.exit(0)

    # --- Normal daily mode ---
    mode = get_trading_mode()
    print("=== StoMar Daily Signal Loop ===")
    print(mode_banner(mode))
    if mode.value == "live":
        # Live mode requires the full gate — refuse to proceed otherwise.
        require_live_allowed()
        print("LIVE MODE: orders will be submitted to a real broker.")
    print(f"Date:     {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}")
    print(f"Tickers:  {len(tickers)}")
    print(f"Ledger:   {args.db}")
    print(f"Dry run:  {args.dry_run}")
    print(f"Paper:    {args.paper_trade}")
    print()

    meta_controller = MetaController()

    # Try loading pre-trained meta-controller (manifest-verified)
    if args.train_meta:
        print("Training meta-controller on ledger history...")
        result = meta_controller.train(ledger)
        print(f"  Status: {result['status']}")
        if result["status"] == "trained":
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Samples:  {result['n_samples']}")
            weights = meta_controller.get_weights()
            print("  Top signals:")
            for name, weight in list(weights.items())[:5]:
                direction = "positive" if weight > 0 else "negative"
                print(f"    {name}: {weight:+.4f} ({direction})")
            meta_controller.save()
            print(f"  Saved to {META_CONTROLLER_PATH}")
    elif os.path.exists(META_CONTROLLER_PATH):
        if meta_controller.load():
            print("Loaded pre-trained meta-controller.")
        else:
            print("WARNING: meta-controller artifact failed verification; running rule-based.")
    print()

    orchestrator = DailyOrchestrator(
        tickers=tickers,
        ledger=ledger,
        meta_controller=meta_controller,
    )

    if args.schedule_hours > 0:
        while not shutdown_requested:
            summary = orchestrator.run(dry_run=args.dry_run)
            print("\n=== Summary ===")
            print(f"Decisions: {len(summary['decisions'])}")
            print(f"Errors:    {len(summary['errors'])}")

            for d in summary["decisions"]:
                print(f"  [{d['action']:4s}] {d['ticker']:15s} "
                      f"size={d['position_size']:.2%} conf={d['confidence']:.2f}")

            for e in summary["errors"]:
                print(f"  [ERR]  {e['ticker']:15s} {e['error']}")

            if shutdown_requested:
                break
            if args.paper_trade and summary["decisions"]:
                run_paper_trades(summary["decisions"], ledger, capital=args.capital)

            if shutdown_requested:
                break
            ledger.close()
            print(f"\nSleeping for {args.schedule_hours:.2f} hours...")
            # Sleep in small increments so we can react to shutdown signals
            sleep_end = time.time() + args.schedule_hours * 3600
            while time.time() < sleep_end and not shutdown_requested:
                time.sleep(1)
            ledger = Ledger(args.db)  # Re-open after sleep

    summary = orchestrator.run(dry_run=args.dry_run)

    print("\n=== Summary ===")
    print(f"Decisions: {len(summary['decisions'])}")
    print(f"Errors:    {len(summary['errors'])}")

    for d in summary["decisions"]:
        print(f"  [{d['action']:4s}] {d['ticker']:15s} "
              f"size={d['position_size']:.2%} conf={d['confidence']:.2f}")

    for e in summary["errors"]:
        print(f"  [ERR]  {e['ticker']:15s} {e['error']}")

    # Auto-execute paper trades if requested
    if not shutdown_requested and args.paper_trade and summary["decisions"]:
        run_paper_trades(summary["decisions"], ledger, capital=args.capital)

    # Auto-cleanup after successful daily loop
    if not shutdown_requested and (args.cleanup or not args.dry_run):
        try:
            from src.data.data_retention import DataRetentionPolicy, print_report
            policy = DataRetentionPolicy()
            cleanup_report = policy.run_all(active_tickers=tickers, dry_run=False)
            if any(v for k, v in cleanup_report.items() if k != "timestamp" and k != "dry_run" and k != "disk_usage_before" and isinstance(v, (int, dict))):
                print_report(cleanup_report)
        except Exception as e:
            logging.warning("Data retention cleanup failed: %s", e)

    ledger.close()
    sys.exit(0 if not summary["errors"] else 1)


if __name__ == "__main__":
    main()

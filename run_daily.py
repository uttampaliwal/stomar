"""Daily autonomous loop. Run via scheduler or manually.

Usage:
    python run_daily.py                  # Run for all NSE stocks
    python run_daily.py --ticker RELIANCE.NS TCS.NS  # Specific tickers
    python run_daily.py --dry-run        # Run but don't log to ledger
    python run_daily.py --db stomar.db   # Custom ledger path
"""

import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import NSE_STOCKS
from src.ledger import Ledger
from src.orchestrator import DailyOrchestrator
from src.meta_controller import MetaController


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    parser = argparse.ArgumentParser(description="StoMar Daily Signal Loop")
    parser.add_argument("--ticker", nargs="+", metavar="TICKER",
                        help="Tickers to process (default: all NSE stocks)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run signals but don't write to ledger")
    parser.add_argument("--db", default="stomar.db",
                        help="SQLite ledger path (default: stomar.db)")
    parser.add_argument("--train-meta", action="store_true",
                        help="Train meta-controller on ledger history before running")
    args = parser.parse_args()

    tickers = args.ticker if args.ticker else NSE_STOCKS
    print(f"=== StoMar Daily Signal Loop ===")
    print(f"Date:     {__import__('datetime').datetime.now().strftime('%Y-%m-%d')}")
    print(f"Tickers:  {len(tickers)}")
    print(f"Ledger:   {args.db}")
    print(f"Dry run:  {args.dry_run}")
    print()

    ledger = Ledger(args.db)
    meta_controller = MetaController()

    if args.train_meta:
        print("Training meta-controller on ledger history...")
        result = meta_controller.train(ledger)
        print(f"  Status: {result['status']}")
        if result["status"] == "trained":
            print(f"  Accuracy: {result['accuracy']:.1%}")
            print(f"  Samples: {result['n_samples']}")
            weights = meta_controller.get_weights()
            print("  Top signals:")
            for name, weight in list(weights.items())[:5]:
                direction = "positive" if weight > 0 else "negative"
                print(f"    {name}: {weight:+.4f} ({direction})")
        print()

    orchestrator = DailyOrchestrator(
        tickers=tickers,
        ledger=ledger,
        meta_controller=meta_controller,
    )

    summary = orchestrator.run(dry_run=args.dry_run)

    print(f"\n=== Summary ===")
    print(f"Decisions: {len(summary['decisions'])}")
    print(f"Errors:    {len(summary['errors'])}")

    for d in summary["decisions"]:
        print(f"  [{d['action']:4s}] {d['ticker']:15s} "
              f"size={d['position_size']:.2%} conf={d['confidence']:.2f}")

    for e in summary["errors"]:
        print(f"  [ERR]  {e['ticker']:15s} {e['error']}")

    ledger.close()
    sys.exit(0 if not summary["errors"] else 1)


if __name__ == "__main__":
    main()

"""Read-only Kite Connect sandbox validation.

Usage:
    python scripts/validate_kite_sandbox.py                 # live gate must pass
    python scripts/validate_kite_sandbox.py --ticker TCS.NS INFY.NS

The script performs READ-ONLY checks only (margins, instrument mapping,
positions). It NEVER places, modifies, or cancels an order. Run it against
the Zerodha sandbox before any real account is ever used.

Exit code 0 = all checks passed; 1 = any check failed.
"""

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.brokers import get_broker
from src.core.trading_mode import LiveTradingNotEnabledError


def main() -> int:
    parser = argparse.ArgumentParser(description="Read-only Kite sandbox validation")
    parser.add_argument("--ticker", nargs="+", default=None,
                        help="Tickers to resolve (default: RELIANCE.NS TCS.NS HDFCBANK.NS)")
    args = parser.parse_args()

    try:
        broker = get_broker(force_live=True)
    except (LiveTradingNotEnabledError, ImportError, ValueError) as exc:
        print(f"live broker unavailable: {exc}")
        print("set STOMAR_LIVE_TRADING=true, STOMAR_LIVE_ACCOUNT_APPROVED=true,")
        print("STOMAR_LIVE_CONFIRMATION=I_CONFIRM_REAL_MONEY_TRADING, and broker")
        print("credentials, or use sandbox credentials.")
        return 1

    if not broker.is_live:
        print("ERROR: expected a live broker, got a dry-run broker.")
        return 1

    try:
        cash = broker.get_margin().get("available_cash", "?")
    except Exception as exc:
        cash = f"<fetch failed: {exc}>"
    print(f"validating live broker: {broker.name} (account {cash} cash)")
    report = broker.validate_sandbox(tickers=args.ticker)
    print(json.dumps(report, indent=2))

    ok = report.get("ok", False)
    print("\nRESULT:", "PASS" if ok else "FAIL")
    if ok:
        print("read-only validation passed; a supervised paper dry-run of the")
        print("broker-backed execution path is the next step.")
    else:
        print("fix the failed checks before any live order is attempted.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

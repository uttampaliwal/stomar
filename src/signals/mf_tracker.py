"""Mutual Fund portfolio tracker.

Tracks MF holdings, fetches NAV history, computes XIRR, and estimates
factor exposures from return-based analysis.

Usage:
    from src.signals.mf_tracker import MFTracker
    tracker = MFTracker()
    tracker.add_holding("0P0000OQN8", units=100, avg_nav=150.0, fund_name="HDFC Flex Cap")
    tracker.fetch_nav_history("0P0000OQN8", period="2y")
    print(tracker.get_portfolio_value())
"""

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.trading.holdings import INDIAN_MF_MAP

logger = logging.getLogger(__name__)


@dataclass
class MFHolding:
    ticker: str
    units: float
    avg_nav: float
    fund_name: str = ""
    category: str = ""


class MFTracker:
    """Mutual fund portfolio tracker with NAV history and factor analysis."""

    _NAV_CACHE_MAX = 50  # Max cached fund histories

    def __init__(self):
        self.holdings: dict[str, MFHolding] = {}
        self.nav_cache: dict[str, pd.Series] = {}

    def add_holding(self, ticker: str, units: float, avg_nav: float,
                    fund_name: str = "", category: str = ""):
        """Add or update a MF holding."""
        self.holdings[ticker] = MFHolding(
            ticker=ticker, units=units, avg_nav=avg_nav,
            fund_name=fund_name, category=category,
        )

    def remove_holding(self, ticker: str):
        """Remove a holding."""
        self.holdings.pop(ticker, None)
        self.nav_cache.pop(ticker, None)

    def fetch_nav_history(self, ticker: str, period: str = "2y") -> pd.Series:
        """Fetch NAV history from yfinance. Caches in memory."""
        if ticker in self.nav_cache:
            return self.nav_cache[ticker]

        yf_ticker = INDIAN_MF_MAP.get(ticker, ticker)
        if yf_ticker is None:
            logger.warning("No yfinance ticker for %s", ticker)
            return pd.Series(dtype=float)

        try:
            import yfinance as yf
            data = yf.download(yf_ticker, period=period, progress=False)
            if data.empty:
                logger.warning("No NAV data for %s", ticker)
                return pd.Series(dtype=float)
            nav = data["Close"].dropna()
            if len(self.nav_cache) >= self._NAV_CACHE_MAX:
                oldest = next(iter(self.nav_cache))
                del self.nav_cache[oldest]
            self.nav_cache[ticker] = nav
            return nav
        except Exception as e:
            logger.error("Failed to fetch NAV for %s: %s", ticker, e)
            return pd.Series(dtype=float)

    def get_current_value(self, ticker: str) -> float:
        """Latest NAV × units."""
        if ticker not in self.holdings:
            return 0.0
        holding = self.holdings[ticker]
        nav = self.nav_cache.get(ticker)
        if nav is not None and len(nav) > 0:
            return float(nav.iloc[-1]) * holding.units
        return holding.avg_nav * holding.units

    def get_portfolio_value(self) -> float:
        """Total value across all holdings."""
        return sum(self.get_current_value(t) for t in self.holdings)

    def get_invested_value(self) -> float:
        """Total invested (avg_nav × units)."""
        return sum(h.avg_nav * h.units for h in self.holdings.values())

    def get_total_pnl(self) -> float:
        """Total P&L across all holdings."""
        return self.get_portfolio_value() - self.get_invested_value()

    def get_total_return_pct(self) -> float:
        """Total return percentage."""
        invested = self.get_invested_value()
        if invested == 0:
            return 0.0
        return self.get_total_pnl() / invested

    def compute_xirr(self, ticker: str = None) -> float:
        """XIRR for a specific fund or entire portfolio.

        Uses annualized return approximation when full cashflow data isn't available.
        """
        if ticker is not None:
            return self._compute_single_xirr(ticker)
        return self._compute_portfolio_xirr()

    def _compute_single_xirr(self, ticker: str) -> float:
        """XIRR for a single holding."""
        if ticker not in self.holdings:
            return 0.0
        holding = self.holdings[ticker]
        nav = self.nav_cache.get(ticker)
        if nav is None or len(nav) < 2:
            return 0.0

        current_nav = float(nav.iloc[-1])
        cost_basis = holding.avg_nav
        if cost_basis <= 0:
            return 0.0

        total_return = (current_nav / cost_basis) - 1
        # Approximate annualization from available data
        n_days = len(nav)
        n_years = max(n_days / 252, 0.01)
        try:
            ann_return = (1 + total_return) ** (1 / n_years) - 1
            return ann_return
        except (ValueError, ZeroDivisionError):
            return 0.0

    def _compute_portfolio_xirr(self) -> float:
        """Approximate portfolio-level XIRR."""
        total_invested = self.get_invested_value()
        total_current = self.get_portfolio_value()
        if total_invested <= 0:
            return 0.0

        total_return = (total_current / total_invested) - 1
        # Use longest NAV history for annualization
        max_days = 0
        for ticker in self.holdings:
            nav = self.nav_cache.get(ticker)
            if nav is not None:
                max_days = max(max_days, len(nav))
        n_years = max(max_days / 252, 0.01) if max_days > 0 else 1.0

        try:
            return (1 + total_return) ** (1 / n_years) - 1
        except (ValueError, ZeroDivisionError):
            return 0.0

    def compute_factor_exposures(self, ticker: str) -> dict:
        """Estimate value/momentum/quality factor loadings from returns.

        Simple regression of fund returns against factor proxies.
        """
        nav = self.nav_cache.get(ticker)
        if nav is None or len(nav) < 60:
            return {"value": 0.0, "momentum": 0.0, "quality": 0.0}

        returns = nav.pct_change().dropna()

        # Momentum: 12-month return (skip last month)
        if len(returns) > 252:
            momentum_12m = float(returns.iloc[-252:-21].sum())
        elif len(returns) > 60:
            momentum_12m = float(returns.sum())
        else:
            momentum_12m = 0.0

        # Volatility (inverse of quality proxy)
        vol = float(returns.std()) * np.sqrt(252) if len(returns) > 20 else 0.2

        # Simple factor estimates based on return characteristics
        value_proxy = -momentum_12m if momentum_12m != 0 else 0  # contrarian
        quality_proxy = 1.0 / max(vol, 0.01)  # low vol = high quality

        # Normalize to [-1, 1]
        def normalize(x, scale=0.5):
            return max(-1.0, min(1.0, x / scale))

        return {
            "value": normalize(value_proxy),
            "momentum": normalize(momentum_12m),
            "quality": normalize(quality_proxy - 2.0),
            "volatility": vol,
        }

    def get_top_holdings(self, n: int = 5) -> list[dict]:
        """Top N holdings by current value."""
        values = []
        for ticker, holding in self.holdings.items():
            current = self.get_current_value(ticker)
            values.append({
                "ticker": ticker,
                "fund_name": holding.fund_name,
                "units": holding.units,
                "avg_nav": holding.avg_nav,
                "current_value": current,
                "pnl": current - (holding.avg_nav * holding.units),
                "return_pct": ((current / (holding.avg_nav * holding.units)) - 1)
                if holding.avg_nav > 0 else 0,
            })
        values.sort(key=lambda x: x["current_value"], reverse=True)
        return values[:n]

    def get_allocation_breakdown(self) -> dict:
        """Category-level allocation."""
        total = self.get_portfolio_value()
        if total <= 0:
            return {}

        categories = {}
        for holding in self.holdings.values():
            cat = holding.category or "Unknown"
            if cat not in categories:
                categories[cat] = 0.0
            ticker = holding.ticker
            categories[cat] += self.get_current_value(ticker)

        return {cat: val / total for cat, val in categories.items()}

    def detect_concentration_risk(self, threshold: float = 0.3) -> list[dict]:
        """Flag holdings exceeding threshold."""
        total = self.get_portfolio_value()
        if total <= 0:
            return []

        flagged = []
        for ticker, holding in self.holdings.items():
            weight = self.get_current_value(ticker) / total
            if weight > threshold:
                flagged.append({
                    "ticker": ticker,
                    "fund_name": holding.fund_name,
                    "weight": weight,
                    "threshold": threshold,
                })
        return flagged

    def compare_to_benchmark(self, ticker: str, benchmark: str = "^NSEI") -> dict:
        """Compare fund returns vs Nifty 50."""
        nav = self.nav_cache.get(ticker)
        if nav is None or len(nav) < 20:
            return {"error": "Insufficient data"}

        try:
            import yfinance as yf
            bench = yf.download(benchmark, period="2y", progress=False)["Close"]
            if bench.empty:
                return {"error": "Benchmark data unavailable"}
        except Exception:
            return {"error": "Failed to fetch benchmark"}

        # Align dates
        common_dates = nav.index.intersection(bench.index)
        if len(common_dates) < 20:
            return {"error": "Insufficient overlapping dates"}

        fund_ret = nav.loc[common_dates].pct_change().dropna()
        bench_ret = bench.loc[common_dates].pct_change().dropna()

        fund_annual = float(fund_ret.mean() * 252)
        bench_annual = float(bench_ret.mean() * 252)
        fund_vol = float(fund_ret.std() * np.sqrt(252))
        bench_vol = float(bench_ret.std() * np.sqrt(252))

        tracking_error = float((fund_ret - bench_ret).std() * np.sqrt(252))
        info_ratio = (fund_annual - bench_annual) / tracking_error if tracking_error > 0 else 0

        return {
            "fund_annual_return": fund_annual,
            "benchmark_annual_return": bench_annual,
            "fund_volatility": fund_vol,
            "benchmark_volatility": bench_vol,
            "excess_return": fund_annual - bench_annual,
            "tracking_error": tracking_error,
            "information_ratio": info_ratio,
            "correlation": float(fund_ret.corr(bench_ret)),
        }

    def save_state(self, path: str = None):
        """Persist holdings + cached NAVs."""
        if path is None:
            from src.core.constants import MF_STATE_PATH
            path = MF_STATE_PATH
        state = {
            "holdings": {
                t: {"units": h.units, "avg_nav": h.avg_nav,
                    "fund_name": h.fund_name, "category": h.category}
                for t, h in self.holdings.items()
            },
            "nav_cache": {
                t: {str(k): v for k, v in nav.to_dict().items()}
                for t, nav in self.nav_cache.items()
            },
        }
        with open(path, "w") as f:
            json.dump(state, f, indent=2, default=str)
        logger.info("MF state saved to %s", path)

    def load_state(self, path: str = None) -> bool:
        """Load from disk."""
        import os
        if path is None:
            from src.core.constants import MF_STATE_PATH
            path = MF_STATE_PATH
        if not os.path.exists(path):
            return False
        with open(path) as f:
            state = json.load(f)

        self.holdings = {
            t: MFHolding(ticker=t, **v)
            for t, v in state.get("holdings", {}).items()
        }
        self.nav_cache = {
            t: pd.Series(v, index=pd.to_datetime(list(v.keys())))
            for t, v in state.get("nav_cache", {}).items()
        }
        logger.info("MF state loaded from %s", path)
        return True

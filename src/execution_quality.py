"""Execution quality analysis.

Compares fill prices against benchmarks (VWAP, open, close) to measure
how well trades are executed in paper trading mode.

Usage:
    from src.execution_quality import ExecutionQualityAnalyzer
    eq = ExecutionQualityAnalyzer()
    eq.add_fill("RELIANCE.NS", "BUY", 2500, 10, 2505)
    eq.set_vwap("RELIANCE.NS", 2503)
    report = eq.analyze()
"""

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    ticker: str
    side: str
    intended_price: float
    filled_price: float
    quantity: int
    fill_cost: float = 0.0
    slippage: float = 0.0


class ExecutionQualityAnalyzer:
    """Measures execution quality against benchmarks."""

    def __init__(self):
        self.fills: list[FillRecord] = []
        self.vwap_benchmarks: dict[str, float] = {}
        self.open_benchmarks: dict[str, float] = {}
        self.close_benchmarks: dict[str, float] = {}

    def add_fill(self, ticker: str, side: str, intended_price: float,
                 quantity: int, filled_price: float):
        slippage = filled_price - intended_price
        if side == "SELL":
            slippage = -slippage

        fill = FillRecord(
            ticker=ticker, side=side,
            intended_price=intended_price, filled_price=filled_price,
            quantity=quantity, fill_cost=filled_price * quantity,
            slippage=slippage,
        )
        self.fills.append(fill)

    def set_vwap(self, ticker: str, vwap: float):
        self.vwap_benchmarks[ticker] = vwap

    def set_open(self, ticker: str, open_price: float):
        self.open_benchmarks[ticker] = open_price

    def set_close(self, ticker: str, close_price: float):
        self.close_benchmarks[ticker] = close_price

    def analyze(self) -> dict:
        """Compute execution quality metrics."""
        if not self.fills:
            return self._empty_report()

        buy_fills = [f for f in self.fills if f.side == "BUY"]
        sell_fills = [f for f in self.fills if f.side == "SELL"]

        buy_slippages = [f.slippage for f in buy_fills]
        sell_slippages = [f.slippage for f in sell_fills]
        all_slippages = [f.slippage for f in self.fills]

        # VWAP slippage
        vwap_slippages = []
        for fill in self.fills:
            if fill.ticker in self.vwap_benchmarks:
                vwap = self.vwap_benchmarks[fill.ticker]
                vs = fill.filled_price - vwap
                if fill.side == "SELL":
                    vs = -vs
                vwap_slippages.append(vs)

        # Open slippage
        open_slippages = []
        for fill in self.fills:
            if fill.ticker in self.open_benchmarks:
                op = self.open_benchmarks[fill.ticker]
                os_val = fill.filled_price - op
                if fill.side == "SELL":
                    os_val = -os_val
                open_slippages.append(os_val)

        total_volume = sum(f.quantity for f in self.fills)
        total_value = sum(f.fill_cost for f in self.fills)

        return {
            "total_fills": len(self.fills),
            "total_volume": total_volume,
            "total_value": total_value,
            "avg_slippage": _safe_mean(all_slippages),
            "avg_buy_slippage": _safe_mean(buy_slippages),
            "avg_sell_slippage": _safe_mean(sell_slippages),
            "max_slippage": max(all_slippages) if all_slippages else 0,
            "min_slippage": min(all_slippages) if all_slippages else 0,
            "vwap_slippage": _safe_mean(vwap_slippages),
            "open_slippage": _safe_mean(open_slippages),
            "slippage_bps": _safe_mean(all_slippages) / max(total_value / max(total_volume, 1), 1) * 10000
            if total_volume > 0 else 0,
            "fill_rate": 1.0,
            "buy_count": len(buy_fills),
            "sell_count": len(sell_fills),
        }

    def per_ticker(self) -> dict[str, dict]:
        """Per-ticker execution quality."""
        tickers = set(f.ticker for f in self.fills)
        result = {}
        for ticker in tickers:
            tf = [f for f in self.fills if f.ticker == ticker]
            slippages = [f.slippage for f in tf]
            vol = sum(f.quantity for f in tf)
            val = sum(f.fill_cost for f in tf)
            result[ticker] = {
                "fills": len(tf),
                "volume": vol,
                "value": val,
                "avg_slippage": _safe_mean(slippages),
            }
        return result

    def _empty_report(self) -> dict:
        return {
            "total_fills": 0, "total_volume": 0, "total_value": 0,
            "avg_slippage": 0, "avg_buy_slippage": 0, "avg_sell_slippage": 0,
            "max_slippage": 0, "min_slippage": 0,
            "vwap_slippage": 0, "open_slippage": 0,
            "slippage_bps": 0, "fill_rate": 1.0,
            "buy_count": 0, "sell_count": 0,
        }


def _safe_mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)

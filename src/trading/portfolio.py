import pandas as pd
import numpy as np
from datetime import datetime
from src.core.constants import (
    BROKERAGE_RATE, RISK_FREE_RATE, calculate_nse_costs,
    STCG_TAX_RATE, LTCG_TAX_RATE, LTCG_EXEMPTION, LONG_TERM_HOLDING_DAYS,
)


class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}
        self.trades = []
        self.taxes_paid = 0.0
        self.ltcg_exemption_remaining = LTCG_EXEMPTION
        self.equity_curve = [{"date": datetime.now(), "equity": initial_capital}]

    def buy(self, ticker, price, quantity, date, brokerage=BROKERAGE_RATE):
        costs = calculate_nse_costs(price, quantity, "buy")
        cost = price * quantity
        total_cost = cost + costs["total"]
        if total_cost > self.cash:
            max_qty = int(self.cash / (price + price * costs["effective_rate"]))
            if max_qty == 0:
                return False
            quantity = max_qty
            costs = calculate_nse_costs(price, quantity, "buy")
            cost = price * quantity
            total_cost = cost + costs["total"]

        self.cash -= total_cost
        # Store cost-inclusive average price for accurate P&L calculation
        cost_incl_per_share = (price * quantity + costs["total"]) / quantity
        if ticker in self.holdings:
            old_qty, old_price, old_date = self.holdings[ticker]
            total_qty = old_qty + quantity
            avg_price = ((old_price * old_qty) + (cost_incl_per_share * quantity)) / total_qty
            self.holdings[ticker] = (total_qty, avg_price, old_date)
        else:
            self.holdings[ticker] = (quantity, cost_incl_per_share, date)

        self.trades.append({
            "date": date, "ticker": ticker, "action": "BUY",
            "price": price, "quantity": quantity,
            "cost": total_cost, "costs": costs,
        })
        self._update_equity(date, prices={ticker: price})
        return True

    def sell(self, ticker, price, quantity, date, brokerage=BROKERAGE_RATE):
        if ticker not in self.holdings:
            return False
        held_qty, avg_price, first_buy_date = self.holdings[ticker]
        quantity = min(quantity, held_qty)
        costs = calculate_nse_costs(price, quantity, "sell")
        proceeds = price * quantity
        net_proceeds = proceeds - costs["total"]
        # P&L: avg_price already includes buy-side costs, so only subtract sell-side costs
        pnl = (price - avg_price) * quantity - costs["total"]

        # P5.5: capital gains tax (STCG 15% < 1yr, LTCG 10% >= 1yr with ₹1L exemption)
        tax = 0.0
        try:
            holding_days = (pd.Timestamp(date) - pd.Timestamp(first_buy_date)).days
        except (TypeError, ValueError):
            holding_days = 0
        if pnl > 0:
            if holding_days >= LONG_TERM_HOLDING_DAYS:
                taxable = max(pnl - self.ltcg_exemption_remaining, 0.0)
                self.ltcg_exemption_remaining = max(self.ltcg_exemption_remaining - pnl, 0.0)
                tax = round(taxable * LTCG_TAX_RATE, 2)
            else:
                tax = round(pnl * STCG_TAX_RATE, 2)
            net_proceeds -= tax
            self.taxes_paid += tax

        self.cash += net_proceeds
        remaining = held_qty - quantity
        if remaining == 0:
            del self.holdings[ticker]
        else:
            self.holdings[ticker] = (remaining, avg_price, first_buy_date)

        self.trades.append({
            "date": date, "ticker": ticker, "action": "SELL",
            "price": price, "quantity": quantity,
            "proceeds": net_proceeds, "costs": costs, "pnl": pnl,
            "tax": round(tax, 2), "holding_days": holding_days,
            "tax_category": "LTCG" if holding_days >= LONG_TERM_HOLDING_DAYS else "STCG",
        })
        self._update_equity(date, prices={ticker: price})
        return True

    def _update_equity(self, date, prices=None):
        total = self.cash
        for ticker, (qty, avg_p, _) in self.holdings.items():
            if prices and ticker in prices:
                total += qty * prices[ticker]
            else:
                total += qty * avg_p
        self.equity_curve.append({"date": date, "equity": total})

    def portfolio_value(self, prices=None):
        total = self.cash
        if prices:
            for ticker, (qty, _, _) in self.holdings.items():
                if ticker in prices:
                    total += qty * prices[ticker]
                else:
                    total += qty * self.holdings[ticker][1]
        else:
            for ticker, (qty, avg_p, _) in self.holdings.items():
                total += qty * avg_p
        return total

    def get_stats(self):
        if len(self.equity_curve) < 2:
            return {}
        equity = pd.Series([e["equity"] for e in self.equity_curve])
        rets = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        daily_rf = RISK_FREE_RATE / 252
        # Standard annualised Sharpe: excess annual return over annual vol.
        # (Previously subtracted daily_rf from mean daily return before
        # scaling, which understated the rf drag by sqrt(252).)
        sharpe = float((rets.mean() * 252 - RISK_FREE_RATE) / (rets.std() * np.sqrt(252))) if rets.std() > 0 else 0
        cummax = equity.cummax()
        drawdown = (equity - cummax) / cummax
        max_dd = float(drawdown.min())
        win_trades = [t for t in self.trades if t.get("pnl", 0) > 0]
        loss_trades = [t for t in self.trades if t.get("pnl", 0) < 0]
        win_rate = len(win_trades) / max(len(self.trades), 1)
        return {
            "total_return": total_return,
            "sharpe_ratio": round(sharpe, 2),
            "max_drawdown": round(max_dd * 100, 1),
            "win_rate": round(win_rate * 100, 1),
            "total_trades": len(self.trades),
            "cash_remaining": round(self.cash, 2),
            "holdings_value": round(self.portfolio_value() - self.cash, 2),
            "taxes_paid": round(self.taxes_paid, 2),
            "post_tax_return": round((self.portfolio_value() - self.taxes_paid - self.initial_capital) / self.initial_capital, 4),
        }

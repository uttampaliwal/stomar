import pandas as pd
import numpy as np
from datetime import datetime
from src.core.constants import BROKERAGE_RATE, calculate_nse_costs


class Portfolio:
    def __init__(self, initial_capital=100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.holdings = {}
        self.trades = []
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
            old_qty, old_price = self.holdings[ticker]
            total_qty = old_qty + quantity
            avg_price = ((old_price * old_qty) + (cost_incl_per_share * quantity)) / total_qty
            self.holdings[ticker] = (total_qty, avg_price)
        else:
            self.holdings[ticker] = (quantity, cost_incl_per_share)

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
        held_qty, avg_price = self.holdings[ticker]
        quantity = min(quantity, held_qty)
        costs = calculate_nse_costs(price, quantity, "sell")
        proceeds = price * quantity
        net_proceeds = proceeds - costs["total"]
        # P&L accounts for buy-side costs embedded in avg_price + sell-side costs
        pnl = (price - avg_price) * quantity - costs["total"]

        self.cash += net_proceeds
        remaining = held_qty - quantity
        if remaining == 0:
            del self.holdings[ticker]
        else:
            self.holdings[ticker] = (remaining, avg_price)

        self.trades.append({
            "date": date, "ticker": ticker, "action": "SELL",
            "price": price, "quantity": quantity,
            "proceeds": net_proceeds, "costs": costs, "pnl": pnl,
        })
        self._update_equity(date, prices={ticker: price})
        return True

    def _update_equity(self, date, prices=None):
        total = self.cash
        for ticker, (qty, avg_p) in self.holdings.items():
            if prices and ticker in prices:
                total += qty * prices[ticker]
            else:
                total += qty * avg_p
        self.equity_curve.append({"date": date, "equity": total})

    def portfolio_value(self, prices=None):
        total = self.cash
        if prices:
            for ticker, (qty, _) in self.holdings.items():
                if ticker in prices:
                    total += qty * prices[ticker]
                else:
                    total += qty * self.holdings[ticker][1]
        else:
            for ticker, (qty, avg_p) in self.holdings.items():
                total += qty * avg_p
        return total

    def get_stats(self):
        if len(self.equity_curve) < 2:
            return {}
        equity = pd.Series([e["equity"] for e in self.equity_curve])
        rets = equity.pct_change().dropna()
        total_return = (equity.iloc[-1] - self.initial_capital) / self.initial_capital
        sharpe = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
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
        }

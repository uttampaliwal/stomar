import pandas as pd
import yfinance as yf
import os

from src.core.constants import DATA_DIR

INDIAN_MF_MAP = {
    "Edelweiss Greater China Equity Offshore Fund": "0P0000SKGZ.BO",
    "Edelweiss US Technology Equity Fund of Fund": "0P0001GLWL.BO",
    "Franklin U.S. Opportunities Equity Active Fund of Funds": "0P0000WLRC.BO",
    "HDFC Balanced Advantage Fund": "0P0000X1MY.BO",
    "HDFC ELSS Tax Saver Fund": "0P0000X1MN.BO",
    "HDFC Flexi Cap Fund": "0P0000X1MJ.BO",
    "HDFC Nifty Top 20 Equal Weight Index Fund": "0P0001I2K6.BO",
    "ICICI Prudential Bharat 22 FOF": "0P0001GUPH.BO",
    "ICICI Prudential Dynamic Asset Allocation Active FOF": "0P0000YVQM.BO",
    "ICICI Prudential Infrastructure Fund": "0P00009LM2.BO",
    "ICICI Prudential Large & Mid Cap Fund": "0P0000YVQ4.BO",
    "ICICI Prudential Short Term Fund": "0P00009LMF.BO",
    "ICICI Prudential Silver ETF FOF": "0P0001IPTQ.BO",
    "Invesco India Contra Fund": "0P00008QXS.BO",
    "JioBlackRock Flexi Cap Fund": None,
    "LIC MF Large & Mid Cap Fund": "0P0001J605.BO",
    "Parag Parikh Flexi Cap Fund": "0P0001AJ6C.BO",
    "Quantum Gold Savings Fund": "0P00007PMV.BO",
    "Quantum Multi Asset Active FoF": "0P0001D3R3.BO",
    "SBI Gold Fund": "0P00009MA3.BO",
    "The Wealth Company Ethical Fund": None,
    "UTI Nifty 50 Index Fund": "0P00007PMW.BO",
}


def parse_holdings_csv(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df = df.rename(columns={
        "Instrument": "name",
        "Qty.": "units",
        "Avg. cost": "avg_nav",
        "LTP": "current_nav",
        "Invested": "invested",
        "Cur. val": "current_value",
        "P&L": "pnl",
        "Net chg.": "net_change_pct",
        "Day chg.": "day_change_pct",
    })
    df = df[~df["name"].isin(["Instrument", ""])]
    df = df.dropna(subset=["name"])
    for col in ["units", "avg_nav", "current_nav", "invested", "current_value", "pnl"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    for col in ["net_change_pct", "day_change_pct"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")
    df = df.dropna(subset=["units", "invested"])
    return df


def compute_xirr(cashflows: list) -> float:
    if len(cashflows) < 2:
        return 0.0

    def npv(rate):
        total = 0
        for cf, days in cashflows:
            years = days / 365.25
            if rate > -1:
                total += cf / ((1 + rate) ** years)
            else:
                total += cf
        return total

    low, high = -0.5, 5.0
    for _ in range(100):
        mid = (low + high) / 2
        if npv(mid) > 0:
            low = mid
        else:
            high = mid
    return mid


def compute_portfolio_stats(holdings_df: pd.DataFrame) -> dict:
    if holdings_df.empty:
        return {}

    total_invested = holdings_df["invested"].sum()
    total_current = holdings_df["current_value"].sum()
    total_pnl = holdings_df["pnl"].sum()
    total_return_pct = (total_pnl / total_invested * 100) if total_invested > 0 else 0

    holdings = []
    for _, row in holdings_df.iterrows():
        weight = row["current_value"] / total_current if total_current > 0 else 0
        ret = row["pnl"] / row["invested"] * 100 if row["invested"] > 0 else 0
        holdings.append({
            "name": row["name"],
            "units": row["units"],
            "avg_nav": row["avg_nav"],
            "current_nav": row.get("current_nav", 0),
            "invested": row["invested"],
            "current_value": row["current_value"],
            "pnl": row["pnl"],
            "return_pct": ret,
            "weight": weight,
            "day_change_pct": row.get("day_change_pct", 0),
            "net_change_pct": row.get("net_change_pct", 0),
        })

    holdings.sort(key=lambda x: x["current_value"], reverse=True)

    categories = {}
    for h in holdings:
        name = h["name"]
        if "Gold" in name:
            cat = "Gold"
        elif "ELSS" in name or "Tax" in name:
            cat = "ELSS"
        elif "Index" in name or "Nifty" in name:
            cat = "Index"
        elif "Short Term" in name or "Debt" in name:
            cat = "Debt"
        elif "US" in name or "China" in name or "Offshore" in name or "Global" in name:
            cat = "International"
        elif "Flexi" in name:
            cat = "Flexi Cap"
        elif "Large" in name and "Mid" in name:
            cat = "Large & Mid Cap"
        elif "Large" in name:
            cat = "Large Cap"
        elif "Mid" in name:
            cat = "Mid Cap"
        elif "Silver" in name:
            cat = "Silver"
        elif "Multi Asset" in name or "Balanced" in name or "Dynamic" in name:
            cat = "Multi Asset"
        elif "Infrastructure" in name:
            cat = "Thematic"
        elif "Contra" in name:
            cat = "Contra"
        elif "Ethical" in name:
            cat = "ESG"
        else:
            cat = "Other"
        if cat not in categories:
            categories[cat] = {"value": 0, "invested": 0, "pnl": 0, "count": 0}
        categories[cat]["value"] += h["current_value"]
        categories[cat]["invested"] += h["invested"]
        categories[cat]["pnl"] += h["pnl"]
        categories[cat]["count"] += 1

    for cat in categories:
        cat_total = categories[cat]["value"]
        categories[cat]["weight"] = cat_total / total_current if total_current > 0 else 0
        categories[cat]["return_pct"] = categories[cat]["pnl"] / categories[cat]["invested"] * 100 if categories[cat]["invested"] > 0 else 0

    return {
        "total_invested": total_invested,
        "total_current": total_current,
        "total_pnl": total_pnl,
        "total_return_pct": total_return_pct,
        "n_holdings": len(holdings),
        "holdings": holdings,
        "categories": categories,
    }


def fetch_mf_performance(holdings_df: pd.DataFrame, period: str = "1y") -> dict:
    performance = {}
    for _, row in holdings_df.iterrows():
        name = row["name"]
        ticker = INDIAN_MF_MAP.get(name)
        if not ticker:
            continue
        try:
            data = yf.download(ticker, period=period, progress=False)
            if data is not None and len(data) > 20:
                close = data["Close"].values
                ret = (close[-1] / close[0] - 1) * 100 if close[0] != 0 else 0
                performance[name] = {
                    "ticker": ticker,
                    "period_return": float(ret),
                    "latest_nav": float(close[-1]),
                    "data_points": len(close),
                }
        except Exception:
            continue
    return performance


def save_holdings(holdings_df: pd.DataFrame, name: str = "default"):
    filepath = os.path.join(DATA_DIR, f"holdings_{name}.csv")
    holdings_df.to_csv(filepath, index=False)
    return filepath


def load_holdings(name: str = "default") -> pd.DataFrame:
    filepath = os.path.join(DATA_DIR, f"holdings_{name}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()

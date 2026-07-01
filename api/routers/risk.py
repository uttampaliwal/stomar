"""Risk analysis endpoint."""

import numpy as np
from fastapi import APIRouter, Query
from src.data.data_fetcher import fetch_stock_data, NSE_STOCKS
from src.trading.risk import generate_risk_report, kelly_criterion
from src.signals.regime import detect_regime
from api.utils import parallel_fetch

router = APIRouter()


@router.get("/{ticker}")
def risk_analysis(ticker: str):
    try:
        df = fetch_stock_data(ticker)
        if df is None or df.empty:
            return {"error": f"No data for {ticker}"}

        returns = df["close"].pct_change().dropna().values
        equity_curve = (1 + df["close"].pct_change().fillna(0)).cumprod().values * 100000

        report = generate_risk_report(returns, equity_curve)
        regime = detect_regime(df["close"])

        wins = returns[returns > 0]
        losses = returns[returns < 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        avg_win = float(wins.mean()) if len(wins) > 0 else 0
        avg_loss = float(abs(losses.mean())) if len(losses) > 0 else 0
        kelly = kelly_criterion(win_rate, avg_win, avg_loss)

        return {
            "ticker": ticker,
            "report": report,
            "regime": regime,
            "kelly": {
                "optimal_fraction": round(kelly, 4),
                "win_rate": round(win_rate, 4),
                "avg_win": round(avg_win, 6),
                "avg_loss": round(avg_loss, 6),
            },
        }
    except Exception as e:
        return {"error": str(e)}


def _fetch_returns(ticker):
    df = fetch_stock_data(ticker, period="1y")
    if df is not None and len(df) > 30:
        return df["close"].pct_change().dropna().values
    return None


@router.get("/portfolio/all")
def portfolio_risk():
    try:
        raw = parallel_fetch(_fetch_returns, NSE_STOCKS, max_workers=8)
        stock_returns = {k: v for k, v in raw.items() if v is not None}

        if not stock_returns:
            return {"error": "No data available"}

        result = {}
        for ticker, returns in stock_returns.items():
            report = generate_risk_report(returns, (1 + np.array(returns)).cumprod() * 100000)
            result[ticker] = report

        return {"stocks": result}
    except Exception as e:
        return {"error": str(e)}

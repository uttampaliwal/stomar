"""FastAPI backend for StoMar React UI."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import (
    market,
    predictions,
    portfolio,
    backtest,
    scanner,
    consensus,
    sentiment,
    optimizer,
    holdings,
    risk,
    volatility,
    ranking,
    scenarios,
    regime,
    monitoring,
    pipeline,
    correlation,
    paper_trading,
    mf_tracker,
    ledger,
)

app = FastAPI(title="StoMar API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(market.router, prefix="/api/market", tags=["Market"])
app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(portfolio.router, prefix="/api/portfolio", tags=["Portfolio"])
app.include_router(backtest.router, prefix="/api/backtest", tags=["Backtest"])
app.include_router(scanner.router, prefix="/api/scanner", tags=["Scanner"])
app.include_router(consensus.router, prefix="/api/consensus", tags=["Consensus"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["Sentiment"])
app.include_router(optimizer.router, prefix="/api/optimizer", tags=["Optimizer"])
app.include_router(holdings.router, prefix="/api/holdings", tags=["Holdings"])
app.include_router(risk.router, prefix="/api/risk", tags=["Risk"])
app.include_router(volatility.router, prefix="/api/volatility", tags=["Volatility"])
app.include_router(ranking.router, prefix="/api/ranking", tags=["Ranking"])
app.include_router(scenarios.router, prefix="/api/scenarios", tags=["Scenarios"])
app.include_router(regime.router, prefix="/api/regime", tags=["Regime"])
app.include_router(monitoring.router, prefix="/api/monitoring", tags=["Monitoring"])
app.include_router(pipeline.router, prefix="/api/pipeline", tags=["Pipeline"])
app.include_router(correlation.router, prefix="/api/correlation", tags=["Correlation"])
app.include_router(paper_trading.router, prefix="/api/paper-trading", tags=["Paper Trading"])
app.include_router(mf_tracker.router, prefix="/api/mf-tracker", tags=["MF Tracker"])
app.include_router(ledger.router, prefix="/api/ledger", tags=["Ledger"])


@app.get("/api/health")
def health():
    return {"status": "ok", "version": "0.1.0"}

# StoMar React Web UI

## Quick Start

### Prerequisites
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (installs Python 3.13 automatically)
- Node.js 22.22+ (see `.nvmrc`)

### Option 1: One-click startup
```bash
./start_dev.sh        # Linux/macOS
start_dev.bat         # Windows
```

### Option 2: Manual startup

**Backend (FastAPI):**
```bash
uv sync && uv run uvicorn api.main:app --reload --port 8000
```

**Frontend (React + Vite):**
```bash
cd web
npm ci
npm run dev
```

### URLs
| Service | URL |
|---------|-----|
| React UI | http://localhost:5173 |
| FastAPI | http://localhost:8000 |
| API Docs | http://localhost:8000/docs |

## Architecture

```
stomar/
├── api/                    # FastAPI backend
│   ├── main.py            # FastAPI app with CORS
│   └── routers/           # API endpoints per module
│       ├── market.py      # Status, stocks, FII/DII, PCR
│       ├── predictions.py # ML ensemble predictions
│       ├── scanner.py     # Multi-stock scan
│       ├── consensus.py   # 14-signal consensus
│       ├── sentiment.py   # FinBERT sentiment
│       ├── backtest.py    # Walk-forward backtest
│       ├── portfolio.py   # Portfolio analytics
│       ├── optimizer.py   # MVO, Black-Litterman
│       ├── risk.py        # VaR, CVaR, Kelly
│       ├── volatility.py  # GARCH, Parkinson, YZ
│       ├── ranking.py     # Cross-sectional ranking
│       ├── scenarios.py   # Strategy comparison
│       ├── regime.py      # Bull/Bear/Sideways
│       ├── correlation.py # Correlation matrix
│       ├── monitoring.py  # System health
│       ├── pipeline.py    # Training pipeline
│       ├── paper_trading.py # Simulated execution
│       ├── mf_tracker.py  # Mutual fund tracking
│       ├── holdings.py    # Zerodha CSV parser
│       ├── ledger.py      # Trading journal
│       ├── insights.py    # Recommendations + enrichment
│       ├── automation.py  # Automation run + decisions
│       └── wealth.py      # Wealth strategies + Monte-Carlo
│
├── web/                    # React frontend
│   ├── src/
│   │   ├── components/    # UI components (Sidebar, Cards, etc.)
│   │   ├── pages/         # All 21 module pages
│   │   ├── hooks/         # useApi, useDebouncedValue
│   │   └── lib/           # Utilities + typed API shapes (api-types.ts)
│   ├── .nvmrc             # Node 22
│   ├── tailwind.config.js
│   └── vite.config.ts     # Proxy /api to :8000
```

## Design System

- **Dark/Light theme** toggle built-in
- **Glassmorphism** cards with backdrop blur
- **Color palette**: Cyan (#22d3ee), Emerald (#10b981), Rose (#f43f5e), Amber (#f59e0b), Violet (#8b5cf6)
- **Typography**: Inter (UI) + JetBrains Mono (data/numbers)
- **Responsive**: Collapsible sidebar, adaptive grids

## All 21 Pages

| # | Page | Description |
|---|------|-------------|
| 1 | Dashboard | Command center overview |
| 2 | Predictions | 5-model ML ensemble signals |
| 3 | Portfolio | Current holdings + allocation |
| 4 | Scanner | Multi-stock BUY/SELL scan |
| 5 | Consensus | 14-signal unified view |
| 6 | Market Pulse | FII/DII flows, options PCR |
| 7 | Sentiment | FinBERT multi-source |
| 8 | Backtest | Walk-forward validation |
| 9 | Optimizer | Portfolio allocation |
| 10 | Risk | VaR, CVaR, Kelly |
| 11 | Volatility | GARCH, Parkinson, YZ |
| 12 | Ranking | Cross-sectional ranking |
| 13 | Scenarios | Strategy comparison |
| 14 | Regime | Bull/Bear/Sideways |
| 15 | Correlation | Cross-asset heatmap |
| 16 | Monitoring | System health |
| 17 | Pipeline | Training status |
| 18 | Paper Trading | Simulated execution |
| 19 | MF Tracker | Mutual fund XIRR |
| 20 | Ledger | Trading journal |
| 21 | Wealth Goals | Monte-Carlo projection + advisor |

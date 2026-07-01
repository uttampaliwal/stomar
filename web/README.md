# StoMar React Web UI

## Quick Start

### Prerequisites
- Python 3.11+ with all dependencies installed
- Node.js 18+ and npm

### Option 1: One-click startup (Windows)
```bash
start_dev.bat
```

### Option 2: Manual startup

**Backend (FastAPI):**
```bash
pip install fastapi uvicorn[standard]
python -m uvicorn api.main:app --reload --port 8000
```

**Frontend (React + Vite):**
```bash
cd web
npm install
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
│       └── ledger.py      # Trading journal
│
├── web/                    # React frontend
│   ├── src/
│   │   ├── components/    # UI components (Sidebar, Cards, etc.)
│   │   ├── pages/         # All 19 module pages
│   │   ├── hooks/         # useApi hook for data fetching
│   │   └── lib/           # Utilities (formatting, cn)
│   ├── tailwind.config.js
│   └── vite.config.ts     # Proxy /api to :8000
│
├── src/                    # Python source
├── api/                    # FastAPI backend
└── start_dev.bat           # One-click startup
```

## Design System

- **Dark/Light theme** toggle built-in
- **Glassmorphism** cards with backdrop blur
- **Color palette**: Cyan (#22d3ee), Emerald (#10b981), Rose (#f43f5e), Amber (#f59e0b), Violet (#8b5cf6)
- **Typography**: Inter (UI) + JetBrains Mono (data/numbers)
- **Responsive**: Collapsible sidebar, adaptive grids

## All 20 Modules

| # | Page | Description |
|---|------|-------------|
| 1 | Dashboard | Command center overview |
| 2 | Predictions | 5-model ML ensemble signals |
| 3 | Scanner | Multi-stock BUY/SELL scan |
| 4 | Consensus | 14-signal unified view |
| 5 | Market Pulse | FII/DII flows, options PCR |
| 6 | Sentiment | FinBERT multi-source |
| 7 | Backtest | Walk-forward validation |
| 8 | Optimizer | Portfolio allocation |
| 9 | Risk | VaR, CVaR, Kelly |
| 10 | Volatility | GARCH, Parkinson, YZ |
| 11 | Ranking | Cross-sectional ranking |
| 12 | Scenarios | Strategy comparison |
| 13 | Regime | Bull/Bear/Sideways |
| 14 | Correlation | Cross-asset heatmap |
| 15 | Monitoring | System health |
| 16 | Pipeline | Training status |
| 17 | Paper Trading | Simulated execution |
| 18 | MF Tracker | Mutual fund XIRR |
| 19 | Ledger | Trading journal |

import { useState } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { Sidebar } from '@/components/Sidebar'
import { ThemeProvider } from '@/components/ThemeProvider'

import Dashboard from '@/pages/Dashboard'
import Predictions from '@/pages/Predictions'
import Scanner from '@/pages/Scanner'
import Consensus from '@/pages/Consensus'
import MarketPulse from '@/pages/MarketPulse'
import Sentiment from '@/pages/Sentiment'
import Backtest from '@/pages/Backtest'
import Optimizer from '@/pages/Optimizer'
import Risk from '@/pages/Risk'
import Volatility from '@/pages/Volatility'
import Ranking from '@/pages/Ranking'
import Scenarios from '@/pages/Scenarios'
import Regime from '@/pages/Regime'
import Correlation from '@/pages/Correlation'
import Monitoring from '@/pages/Monitoring'
import Pipeline from '@/pages/Pipeline'
import PaperTrading from '@/pages/PaperTrading'
import MFTracker from '@/pages/MFTracker'
import Ledger from '@/pages/Ledger'
import Portfolio from '@/pages/Portfolio'

export default function App() {
  const [collapsed, setCollapsed] = useState(false)

  return (
    <ThemeProvider>
      <BrowserRouter>
        <div className="min-h-screen bg-background">
          <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />
          <main className={`transition-all duration-300 ${collapsed ? 'ml-16' : 'ml-56'} p-6`}>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predictions" element={<Predictions />} />
              <Route path="/scanner" element={<Scanner />} />
              <Route path="/consensus" element={<Consensus />} />
              <Route path="/market-pulse" element={<MarketPulse />} />
              <Route path="/sentiment" element={<Sentiment />} />
              <Route path="/backtest" element={<Backtest />} />
              <Route path="/optimizer" element={<Optimizer />} />
              <Route path="/risk" element={<Risk />} />
              <Route path="/volatility" element={<Volatility />} />
              <Route path="/ranking" element={<Ranking />} />
              <Route path="/scenarios" element={<Scenarios />} />
              <Route path="/regime" element={<Regime />} />
              <Route path="/correlation" element={<Correlation />} />
              <Route path="/monitoring" element={<Monitoring />} />
              <Route path="/pipeline" element={<Pipeline />} />
              <Route path="/paper-trading" element={<PaperTrading />} />
              <Route path="/mf-tracker" element={<MFTracker />} />
              <Route path="/ledger" element={<Ledger />} />
              <Route path="/portfolio" element={<Portfolio />} />
            </Routes>
            {/* Footer */}
            <div className="mt-12 border-t border-border pt-4 text-center">
              <p className="text-xs text-muted-foreground">
                StoMar v0.0.3 — 5-Model Ensemble • Walk-Forward Backtest • Black-Litterman Optimizer
              </p>
              <p className="text-[0.6rem] text-muted-foreground mt-1 opacity-50">
                Educational purposes only — not financial advice.
              </p>
            </div>
          </main>
        </div>
      </BrowserRouter>
    </ThemeProvider>
  )
}

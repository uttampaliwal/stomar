import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Scenarios() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const { data, loading, error } = useApi<any>(`/api/scenarios/${ticker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Scenario Analysis</h1>
          <p className="text-sm text-muted-foreground">Strategy comparison: Buy & Hold, Momentum, Mean Reversion, SMA Crossover</p>
        </div>
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
        >
          {stocks.data?.stocks?.map((s: string) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && data.scenarios && (
        <div className="space-y-3">
          {data.scenarios.map((s: any) => (
            <Card key={s.name}>
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm font-semibold">{s.name}</p>
                  <p className="text-xs text-muted-foreground">{s.n_trades} trades</p>
                </div>
                <div className="text-right">
                  <p className={`font-mono text-lg font-bold ${(s.annualized_return || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                    {((s.annualized_return || 0) * 100).toFixed(1)}%
                  </p>
                  <p className="text-xs text-muted-foreground">annualized</p>
                </div>
              </div>
              <div className="grid grid-cols-3 gap-3 mt-3">
                <Stat label="Sharpe" value={(s.sharpe || 0).toFixed(2)} />
                <Stat label="Max DD" value={`${((s.max_drawdown || 0) * 100).toFixed(1)}%`} trend="down" />
                <Stat label="Volatility" value={`${((s.volatility || 0) * 100).toFixed(1)}%`} />
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}

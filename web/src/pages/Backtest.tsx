import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

export default function Backtest() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/backtest/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Backtest</h1>
          <p className="text-sm text-muted-foreground">Walk-forward validation</p>
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

      {data && !data.error && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Accuracy" value={`${((data.ensemble_accuracy || data.accuracy || 0) * 100).toFixed(1)}%`} trend="up" />
            <Stat label="Total Return" value={`${((data.total_return || data.annual_return || 0) * 100).toFixed(1)}%`} />
            <Stat label="Sharpe" value={(data.sharpe || data.sharpe_ratio || 0).toFixed(2)} />
            <Stat label="Win Rate" value={data.win_rate ? `${data.win_rate}%` : (data.total_trades || 0)} />
          </div>

          {/* Monte Carlo */}
          {data.monte_carlo && (
            <>
              <SectionHeader title="Monte Carlo Validation" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Prob Profit" value={`${((data.monte_carlo.prob_profit || 0) * 100).toFixed(1)}%`} />
                <Stat label="Median Return" value={`${((data.monte_carlo.median_return || 0) * 100).toFixed(1)}%`} />
                <Stat label="Worst 5%" value={`${((data.monte_carlo.worst_5pct || 0) * 100).toFixed(1)}%`} trend="down" />
                <Stat label="Max Drawdown" value={`${((data.monte_carlo.max_drawdown || 0) * 100).toFixed(1)}%`} trend="down" />
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

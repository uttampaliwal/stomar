import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatPercent } from '@/lib/utils'

export default function Risk() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const { data, loading, error } = useApi<any>(`/api/risk/${ticker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const report = data?.report
  const regime = data?.regime
  const kelly = data?.kelly

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Risk Analysis</h1>
          <p className="text-sm text-muted-foreground">VaR, CVaR, Sharpe, Kelly criterion</p>
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
          {/* Regime */}
          {regime && (
            <Card className="text-center py-4">
              <Badge variant={regime.regime === 'Bull' ? 'success' : regime.regime === 'Bear' ? 'danger' : 'warning'} className="text-lg px-4 py-1">
                {regime.regime}
              </Badge>
              <p className="text-sm text-muted-foreground mt-2">Confidence: {regime.confidence > 1 ? (regime.confidence || 0).toFixed(1) : ((regime.confidence || 0) * 100).toFixed(1)}%</p>
              {regime.recommendation && (
                <p className="text-xs text-muted-foreground mt-1">
                  {regime.recommendation.action} • {regime.recommendation.allocation} allocation
                </p>
              )}
            </Card>
          )}

          {/* Risk Metrics */}
          {report && (
            <>
              <SectionHeader title="Risk Metrics" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Annual Return" value={formatPercent(report.annual_return || 0)} trend={report.annual_return >= 0 ? 'up' : 'down'} />
                <Stat label="Annual Vol" value={formatPercent(report.annual_volatility || 0)} />
                <Stat label="Sharpe" value={(report.sharpe || 0).toFixed(2)} trend={report.sharpe > 1 ? 'up' : 'neutral'} />
                <Stat label="Sortino" value={(report.sortino || 0).toFixed(2)} />
                <Stat label="Max Drawdown" value={formatPercent(report.max_drawdown || 0)} trend="down" />
                <Stat label="VaR 95%" value={formatPercent(report.var_95 || 0)} trend="down" />
                <Stat label="CVaR 95%" value={formatPercent(report.cvar_95 || 0)} trend="down" />
                <Stat label="Calmar" value={(report.calmar || 0).toFixed(2)} />
              </div>
            </>
          )}

          {/* Kelly Criterion */}
          {kelly && (
            <>
              <SectionHeader title="Kelly Criterion" />
              <Card>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Optimal Fraction</p>
                    <p className="font-mono text-xl font-bold text-cyan">{(kelly.optimal_fraction * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Win Rate</p>
                    <p className="font-mono text-xl font-bold">{(kelly.win_rate * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Avg Win</p>
                    <p className="font-mono text-xl font-bold text-emerald">{(kelly.avg_win * 100).toFixed(2)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Avg Loss</p>
                    <p className="font-mono text-xl font-bold text-rose">{(kelly.avg_loss * 100).toFixed(2)}%</p>
                  </div>
                </div>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}

import { useState } from 'react'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { formatPercent } from '@/lib/utils'
import type { RiskResponse } from '../lib/api-types'

export default function Risk() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<RiskResponse>(`/api/risk/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const report = data?.report
  const regime = data?.regime
  const kelly = data?.kelly

  const drawdownData = report ? [
    { label: 'Max Drawdown', value: report.max_drawdown },
    { label: 'VaR 95%', value: report.var_95 },
    { label: 'CVaR 95%', value: report.cvar_95 },
    { label: 'Best Day', value: report.best_day },
    { label: 'Worst Day', value: report.worst_day },
  ] : []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Risk Analysis"
        description="VaR, CVaR, Sharpe, and Kelly criterion"
        badge="Protected"
      >
        <div className="flex items-center gap-2 rounded-lg border border-border bg-card/70 px-3 py-2">
          <label className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-muted-foreground">Ticker</label>
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="rounded-md border border-border bg-background px-2 py-1 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
          >
            {stocks.data?.stocks?.map((s: string) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
        </div>
      </PageHeader>

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          Portfolio-level regime detection. For per-stock signals, see <span className="text-cyan font-semibold">Scanner</span> or <span className="text-cyan font-semibold">Consensus</span>.
        </p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !('error' in data) && (
        <>
          {regime && (
            <Card className="text-center py-4">
              <Badge variant={regime.regime === 'Bull' ? 'success' : regime.regime === 'Bear' ? 'danger' : 'warning'} className="text-lg px-4 py-1">
                {regime.regime}
              </Badge>
              <p className="text-sm text-muted-foreground mt-2">Confidence: {regime.confidence > 1 ? (regime.confidence || 0).toFixed(1) : ((regime.confidence || 0) * 100).toFixed(1)}%</p>
              {regime.recommendation && (
                <div className="flex items-center justify-center gap-4 mt-2 text-xs text-muted-foreground">
                  <span>{regime.recommendation.action}</span>
                  <span>•</span>
                  <span>{regime.recommendation.allocation}</span>
                  <span>•</span>
                  <span>{regime.recommendation.risk_level}</span>
                </div>
              )}
            </Card>
          )}

          {report && (
            <>
              <SectionHeader title="Risk Metrics" />
              <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
                <Stat label="Annual Return" value={formatPercent(report.annual_return || 0)} trend={report.annual_return >= 0 ? 'up' : 'down'} />
                <Stat label="Annual Vol" value={formatPercent(report.annual_volatility || 0)} />
                <Stat label="Sharpe" value={(report.sharpe || 0).toFixed(2)} trend={report.sharpe > 1 ? 'up' : 'neutral'} />
                <Stat label="Sortino" value={(report.sortino || 0).toFixed(2)} />
                <Stat label="Calmar" value={(report.calmar || 0).toFixed(2)} />
                <Stat label="Skewness" value={(report.skewness || 0).toFixed(3)} />
                <Stat label="Win Days" value={formatPercent(report.positive_days_pct || 0)} />
                <Stat label="Max Drawdown" value={formatPercent(report.max_drawdown || 0)} trend="down" />
                <Stat label="VaR 95%" value={formatPercent(report.var_95 || 0)} trend="down" />
                <Stat label="CVaR 95%" value={formatPercent(report.cvar_95 || 0)} trend="down" />
                <Stat label="VaR 99%" value={formatPercent(report.var_99 || 0)} trend="down" />
                <Stat label="Best Day" value={formatPercent(report.best_day || 0)} trend="up" />
                <Stat label="Worst Day" value={formatPercent(report.worst_day || 0)} trend="down" />
              </div>
            </>
          )}

          {/* Risk Profile Chart */}
          {drawdownData.length > 0 && (
            <>
              <SectionHeader title="Risk Profile" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={200}>
                  <AreaChart data={drawdownData} layout="vertical" margin={{ left: 80, right: 20 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis type="category" dataKey="label" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={80} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                    <Area type="monotone" dataKey="value" fill="hsl(var(--rose) / 0.2)" stroke="hsl(var(--rose))" strokeWidth={1.5} />
                  </AreaChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

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

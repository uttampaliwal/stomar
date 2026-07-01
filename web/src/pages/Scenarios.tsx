import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { formatPercent } from '@/lib/utils'

const COLORS = ['#06b6d4', '#10b981', '#f59e0b', '#f43f5e', '#8b5cf6', '#ec4899', '#14b8a6', '#f97316', '#6366f1', '#84cc16', '#a855f7']

export default function Scenarios() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/scenarios/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const scenarios = data?.scenarios || []
  const chartData = scenarios.map((s: any) => ({
    name: s.name,
    return: (s.annualized_return || 0) * 100,
    sharpe: s.sharpe || 0,
    vol: (s.annualized_vol || 0) * 100,
  }))

  const bestScenario = scenarios.reduce((best: any, s: any) => (s.sharpe || 0) > (best?.sharpe || -Infinity) ? s : best, null)
  const worstScenario = scenarios.reduce((worst: any, s: any) => (s.sharpe || 0) < (worst?.sharpe || Infinity) ? s : worst, null)

  return (
    <div className="space-y-6">
      <PageHeader
        title="Scenario Comparison"
        description="Compare strategy performance across different market conditions"
        badge="Stress"
      >
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
        >
          {stocks.data?.stocks?.map((s: string) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </PageHeader>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error ? (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {bestScenario && (
              <Card className="border-l-4 border-l-emerald bg-emerald/5">
                <p className="text-xs text-muted-foreground uppercase mb-1">Best Scenario</p>
                <p className="font-mono text-lg font-bold text-emerald">{bestScenario.name}</p>
                <p className="text-sm text-muted-foreground">Sharpe: {bestScenario.sharpe?.toFixed(2)} | Return: {formatPercent((bestScenario.annualized_return || 0) * 100)}</p>
              </Card>
            )}
            {worstScenario && (
              <Card className="border-l-4 border-l-rose bg-rose/5">
                <p className="text-xs text-muted-foreground uppercase mb-1">Worst Scenario</p>
                <p className="font-mono text-lg font-bold text-rose">{worstScenario.name}</p>
                <p className="text-sm text-muted-foreground">Sharpe: {worstScenario.sharpe?.toFixed(2)} | Return: {formatPercent((worstScenario.annualized_return || 0) * 100)}</p>
              </Card>
            )}
          </div>

          <SectionHeader title="Sharpe Ratios" />
          <Card className="overflow-hidden">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={chartData} layout="vertical" margin={{ left: 80, right: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={100} />
                <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                <Bar dataKey="sharpe" radius={[0, 4, 4, 0]}>
                  {chartData.map((_: any, i: number) => (
                    <Cell key={i} fill={COLORS[i % COLORS.length]} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <SectionHeader title="Annualized Returns vs Volatility" />
          <Card className="overflow-hidden">
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                <XAxis dataKey="name" tick={{ fontSize: 8, fill: 'hsl(var(--muted-foreground))' }} angle={-45} textAnchor="end" height={60} />
                <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={40} />
                <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                <Bar dataKey="return" fill="#06b6d4" name="Return %" radius={[4, 4, 0, 0]} />
                <Bar dataKey="vol" fill="#f59e0b" name="Vol %" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </Card>

          <SectionHeader title="All Scenarios" />
          <Card className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                  <th className="text-left py-2">Scenario</th>
                  <th className="text-right py-2">Total Return</th>
                  <th className="text-right py-2">Ann. Return</th>
                  <th className="text-right py-2">Ann. Vol</th>
                  <th className="text-right py-2">Sharpe</th>
                  <th className="text-right py-2">Max DD</th>
                  <th className="text-right py-2">Trades</th>
                </tr>
              </thead>
              <tbody>
                {scenarios.map((s: any) => (
                  <tr key={s.name} className="border-b border-border/50 hover:bg-accent/30">
                    <td className="py-2.5 font-mono font-semibold">{s.name}</td>
                    <td className={`py-2.5 text-right font-mono ${(s.total_return || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {formatPercent((s.total_return || 0) * 100)}
                    </td>
                    <td className="py-2.5 text-right font-mono">{formatPercent((s.annualized_return || 0) * 100)}</td>
                    <td className="py-2.5 text-right font-mono">{formatPercent((s.annualized_vol || 0) * 100)}</td>
                    <td className="py-2.5 text-right font-mono">
                      <Badge variant={(s.sharpe || 0) > 0.5 ? 'success' : (s.sharpe || 0) > 0 ? 'default' : 'danger'}>
                        {(s.sharpe || 0).toFixed(2)}
                      </Badge>
                    </td>
                    <td className="py-2.5 text-right font-mono text-rose">{formatPercent((s.max_drawdown || 0) * 100)}</td>
                    <td className="py-2.5 text-right font-mono">{s.n_trades || 0}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </Card>
        </>
      ) : (
        !loading && <EmptyState message="No scenario data is available for this ticker." />
      )}
    </div>
  )
}

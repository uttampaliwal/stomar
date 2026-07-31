import { useState } from 'react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { formatPercent } from '@/lib/utils'

export default function Backtest() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [capital, setCapital] = useState(100000)
  const [stopLoss, setStopLoss] = useState(2)
  const [takeProfit, setTakeProfit] = useState(5)
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/backtest/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const testResults = data?.test_results || []
  const chartData = testResults.map((w: any) => ({
    window: w.window || w.period || '',
    return: ((w.total_return || w.annual_return || 0) * 100),
    trades: w.total_trades || 0,
    accuracy: ((w.ensemble_accuracy || w.accuracy || 0) * 100),
  }))

  return (
    <div className="space-y-6">
      <PageHeader
        title="Walk-Forward Backtest"
        description="Train on 2 years, test on 6 months, rolling forward"
        badge="Historical"
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
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <p className="text-sm text-muted-foreground">
            Train on <span className="font-semibold text-foreground">2 years</span> of data,
            test on <span className="font-semibold text-foreground">6 months</span>,
            then roll forward. This mirrors the same historical validation path used by the CLI runner.
          </p>
          <div className="rounded-lg border border-border bg-background/70 px-3 py-2 font-mono text-xs text-muted-foreground">
            python run_daily.py --backfill --days 252
          </div>
        </div>
      </Card>

      {/* Inputs */}
      <Card>
        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <div>
            <label className="text-xs text-muted-foreground uppercase block mb-1">Starting Capital</label>
            <input type="number" value={capital} onChange={(e) => setCapital(Number(e.target.value))}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground uppercase block mb-1">Stop Loss %</label>
            <input type="number" value={stopLoss} step={0.5} onChange={(e) => setStopLoss(Number(e.target.value))}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan" />
          </div>
          <div>
            <label className="text-xs text-muted-foreground uppercase block mb-1">Take Profit %</label>
            <input type="number" value={takeProfit} step={0.5} onChange={(e) => setTakeProfit(Number(e.target.value))}
              className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan" />
          </div>
        </div>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          <SectionHeader title="Aggregate Metrics" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Ensemble Accuracy" value={`${((data.ensemble_accuracy || 0) * 100).toFixed(1)}%`} trend="up" />
            <Stat label="Total Return" value={formatPercent((data.total_return || 0) * 100)} trend={(data.total_return || 0) >= 0 ? 'up' : 'down'} />
            <Stat label="Sharpe Ratio" value={(data.sharpe || 0).toFixed(2)} trend={(data.sharpe || 0) > 1 ? 'up' : 'neutral'} />
            <Stat label="Sortino" value={(data.sortino || 0).toFixed(2)} />
            <Stat label="Max Drawdown" value={formatPercent((data.max_drawdown || 0) * 100)} trend="down" />
            <Stat label="Total Trades" value={String(data.total_trades || 0)} />
            <Stat label="Win Rate" value={formatPercent(data.win_rate || 0)} />
            <Stat label="Annual Return" value={formatPercent((data.annual_return || 0) * 100)} />
          </div>

          {/* Monte Carlo */}
          {data.monte_carlo && (
            <>
              <SectionHeader title="Monte Carlo Validation" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <Stat label="Prob Profit" value={`${((data.monte_carlo.prob_profit || 0) * 100).toFixed(1)}%`} />
                <Stat label="Median Return" value={formatPercent((data.monte_carlo.median_return || 0) * 100)} />
                <Stat label="Worst 5%" value={formatPercent((data.monte_carlo.worst_5pct || 0) * 100)} trend="down" />
                <Stat label="Max Drawdown" value={formatPercent((data.monte_carlo.max_drawdown || 0) * 100)} trend="down" />
              </div>
            </>
          )}

          {/* Per-Window Breakdown */}
          {chartData.length > 0 && (
            <>
              <SectionHeader title="Per-Window Performance" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={250}>
                  <BarChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="window" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} angle={-45} textAnchor="end" height={60} />
                    <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={40} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                    <Bar dataKey="return" name="Return %" radius={[4, 4, 0, 0]}>
                      {chartData.map((_: any, i: number) => (
                        <Cell key={i} fill={chartData[i].return >= 0 ? '#10b981' : '#f43f5e'} fillOpacity={0.8} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {testResults.length > 0 && (
            <Card className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                    <th className="text-left py-2">Window</th>
                    <th className="text-right py-2">Accuracy</th>
                    <th className="text-right py-2">Total Return</th>
                    <th className="text-right py-2">Sharpe</th>
                    <th className="text-right py-2">Trades</th>
                  </tr>
                </thead>
                <tbody>
                  {testResults.map((w: any, i: number) => (
                    <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2.5 font-mono">{w.window || w.period || `Window ${i + 1}`}</td>
                      <td className="py-2.5 text-right font-mono">
                        <Badge variant={(w.ensemble_accuracy || w.accuracy || 0) >= 0.55 ? 'success' : (w.ensemble_accuracy || w.accuracy || 0) >= 0.5 ? 'default' : 'danger'}>
                          {((w.ensemble_accuracy || w.accuracy || 0) * 100).toFixed(1)}%
                        </Badge>
                      </td>
                      <td className={`py-2.5 text-right font-mono ${(w.total_return || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                        {formatPercent((w.total_return || 0) * 100)}
                      </td>
                      <td className="py-2.5 text-right font-mono">{(w.sharpe || 0).toFixed(2)}</td>
                      <td className="py-2.5 text-right font-mono">{w.total_trades || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

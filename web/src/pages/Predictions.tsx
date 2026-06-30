import { useState, useMemo } from 'react'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell, Area, BarChart
} from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { formatCurrency, formatPercent } from '@/lib/utils'

const DIRECTION_COLORS = {
  UP: 'text-emerald',
  DN: 'text-rose',
  BUY: 'text-emerald',
  SELL: 'text-rose',
}

function computeSMA(data: number[], period: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < period - 1) return null
    const slice = data.slice(i - period + 1, i + 1)
    return slice.reduce((a, b) => a + b, 0) / period
  })
}

export default function Predictions() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [period, setPeriod] = useState('2y')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error, refetch } = useApi<any>(`/api/predictions/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const { data: featureImp } = useApi<any>(`/api/predictions/${debouncedTicker}/feature-importance`)

  const pred = data?.prediction
  const metrics = data?.metrics
  const chartData = data?.chart
  const recentData = data?.recent

  const enrichedChart = useMemo(() => {
    if (!chartData?.length) return []
    const closes = chartData.map((d: any) => d.close)
    const sma20 = computeSMA(closes, 20)
    const sma50 = computeSMA(closes, 50)
    return chartData.map((d: any, i: number) => ({
      ...d,
      date: d.date?.slice(5, 10),
      sma20: sma20[i] ? Math.round(sma20[i]! * 100) / 100 : null,
      sma50: sma50[i] ? Math.round(sma50[i]! * 100) / 100 : null,
      vol: d.volume / 1_000_000,
    }))
  }, [chartData])

  const enrichedRecent = useMemo(() => {
    if (!recentData?.length) return []
    return [...recentData].reverse().map((r: any, i: number, arr: any[]) => {
      const prevClose = i > 0 ? arr[i - 1].close : r.open
      const chgPct = prevClose ? ((r.close - prevClose) / prevClose * 100) : 0
      return { ...r, chgPct: Math.round(chgPct * 100) / 100 }
    })
  }, [recentData])

  return (
    <div className="space-y-6">
      {/* Header with controls */}
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Predictions</h1>
          <p className="text-sm text-muted-foreground">5-model ML ensemble signals</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
          >
            {stocks.data?.stocks?.map((s: string) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <select
            value={period}
            onChange={(e) => setPeriod(e.target.value)}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
          >
            <option value="6mo">6mo</option>
            <option value="1y">1y</option>
            <option value="2y">2y</option>
            <option value="5y">5y</option>
          </select>
          <button
            onClick={refetch}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono hover:bg-accent/50 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          {/* Hero: Prediction + Confidence */}
          <Card className="text-center py-6">
            {pred ? (
              <>
                <div className="flex items-center justify-center gap-3 mb-2">
                  <Badge
                    variant={pred.direction === 'BUY' ? 'success' : 'danger'}
                    className="text-xl px-6 py-1.5"
                  >
                    {pred.direction === 'BUY' ? '▲' : '▼'} {pred.direction}
                  </Badge>
                </div>
                <p className="font-mono text-4xl font-bold mt-2">
                  {((pred.confidence || 0) * 100).toFixed(1)}%
                </p>
                <p className="text-sm text-muted-foreground mt-1">Ensemble Confidence</p>
                {pred.conviction !== undefined && (
                  <p className="text-xs text-muted-foreground mt-1">
                    Conviction: {pred.conviction.toFixed(1)}%
                  </p>
                )}
              </>
            ) : (
              <EmptyState message="Models not trained for this ticker" />
            )}
          </Card>

          {/* Metrics Row */}
          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
              <Stat
                label="Price"
                value={`₹${metrics.price}`}
                sub={metrics.day_change >= 0 ? `+${metrics.day_change} (+${metrics.day_change_pct}%)` : `${metrics.day_change} (${metrics.day_change_pct}%)`}
                trend={metrics.day_change >= 0 ? 'up' : 'down'}
              />
              <Stat label="Day Change" value={`${metrics.day_change_pct >= 0 ? '+' : ''}${metrics.day_change_pct}%`} trend={metrics.day_change_pct >= 0 ? 'up' : 'down'} />
              <Stat
                label="RSI"
                value={metrics.rsi.toFixed(1)}
                sub={metrics.rsi > 70 ? 'Overbought' : metrics.rsi < 30 ? 'Oversold' : 'Neutral'}
                trend={metrics.rsi > 70 ? 'down' : metrics.rsi < 30 ? 'up' : 'neutral'}
              />
              <Stat
                label="MACD"
                value={metrics.macd.toFixed(2)}
                sub={metrics.macd > 0 ? 'Bullish' : 'Bearish'}
                trend={metrics.macd > 0 ? 'up' : 'down'}
              />
              <Stat label="ATR" value={metrics.atr.toFixed(2)} sub="Volatility" />
              <Stat label="Volume" value={`${(metrics.volume / 1_000_000).toFixed(1)}M`} />
              {pred?.details?.ensemble_prob !== undefined && (
                <Stat label="Ensemble Prob" value={`${(pred.details.ensemble_prob * 100).toFixed(1)}%`} />
              )}
            </div>
          )}

          {/* Price Chart */}
          {enrichedChart.length > 0 && (
            <>
              <SectionHeader title="Price Action" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={320}>
                  <ComposedChart data={enrichedChart} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} interval="preserveStartEnd" />
                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={55} />
                    <Tooltip
                      contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '12px' }}
                      labelStyle={{ color: 'hsl(var(--muted-foreground))' }}
                    />
                    <Line type="monotone" dataKey="close" stroke="hsl(var(--foreground))" dot={false} strokeWidth={1.5} name="Close" />
                    <Line type="monotone" dataKey="sma20" stroke="#8b5cf6" dot={false} strokeWidth={1} strokeDasharray="4 4" name="SMA 20" connectNulls />
                    <Line type="monotone" dataKey="sma50" stroke="#f59e0b" dot={false} strokeWidth={1} strokeDasharray="4 4" name="SMA 50" connectNulls />
                    <Area type="monotone" dataKey="vol" fill="hsl(var(--cyan) / 0.15)" stroke="none" name="Volume (M)" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {/* Two-column: Feature Importance + Model Votes */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Feature Importance */}
            {featureImp?.features && (
              <>
                <div>
                  <SectionHeader title="Feature Importance" />
                  <Card>
                    <ResponsiveContainer width="100%" height={280}>
                      <BarChart data={featureImp.features.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={90} />
                        <Tooltip
                          contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(v: number) => [(v * 100).toFixed(2) + '%', 'Importance']}
                        />
                        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                          {featureImp.features.slice(0, 8).map((f: any, i: number) => {
                            const avg = featureImp.features.reduce((a: number, b: any) => a + b.importance, 0) / featureImp.features.length
                            return <Cell key={i} fill={f.importance > avg ? 'hsl(var(--cyan))' : 'hsl(var(--muted))'} />
                          })}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </Card>
                </div>
              </>
            )}

            {/* Model Votes */}
            {pred?.details && (
              <div>
                <SectionHeader title="Model Votes" />
                <Card>
                  <div className="space-y-3">
                    {[
                      { name: 'LSTM', dir: pred.details.lstm_dir, prob: pred.details.lstm_prob },
                      { name: 'GRU', dir: pred.details.gru_dir, prob: pred.details.gru_prob },
                      { name: 'Transformer', dir: pred.details.transformer_dir, prob: pred.details.transformer_prob },
                      { name: 'XGBoost', dir: pred.details.xgb_dir, prob: pred.details.xgb_prob_up },
                      { name: 'LightGBM', dir: pred.details.lgb_dir, prob: pred.details.lgb_prob_up },
                    ].map((m) => {
                      const dirLabel = m.dir === 1 ? 'UP' : 'DN'
                      return (
                        <div key={m.name} className="flex items-center gap-3">
                          <span className="w-24 text-xs font-mono uppercase text-muted-foreground">{m.name}</span>
                          <span className={`w-8 text-xs font-mono font-bold ${m.dir === 1 ? 'text-emerald' : 'text-rose'}`}>
                            {m.dir === 1 ? '▲' : '▼'} {dirLabel}
                          </span>
                          <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${m.dir === 1 ? 'bg-emerald' : 'bg-rose'}`}
                              style={{ width: `${(m.prob || 0) * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs w-12 text-right">{((m.prob || 0) * 100).toFixed(1)}%</span>
                        </div>
                      )
                    })}
                  </div>
                </Card>
              </div>
            )}
          </div>

          {/* Recent Prices */}
          {enrichedRecent.length > 0 && (
            <>
              <SectionHeader title="Recent Prices" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                      <th className="text-left py-2">Date</th>
                      <th className="text-right py-2">Open</th>
                      <th className="text-right py-2">High</th>
                      <th className="text-right py-2">Low</th>
                      <th className="text-right py-2">Close</th>
                      <th className="text-right py-2">Chg%</th>
                      <th className="text-right py-2">Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {enrichedRecent.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                        <td className="py-2 font-mono text-xs">{row.date?.slice(5, 10)}</td>
                        <td className="py-2 text-right font-mono">{row.open}</td>
                        <td className="py-2 text-right font-mono">{row.high}</td>
                        <td className="py-2 text-right font-mono">{row.low}</td>
                        <td className="py-2 text-right font-mono font-semibold">{row.close}</td>
                        <td className={`py-2 text-right font-mono ${row.chgPct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {row.chgPct >= 0 ? '+' : ''}{row.chgPct}%
                        </td>
                        <td className="py-2 text-right font-mono text-muted-foreground">{(row.volume / 1_000_000).toFixed(1)}M</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}

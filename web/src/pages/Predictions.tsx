import { useState, useMemo, useEffect, useCallback } from 'react'
import {
  ComposedChart, Line, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell, Area, LineChart, BarChart
} from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

function computeSMA(data: number[], period: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < period - 1) return null
    const slice = data.slice(i - period + 1, i + 1)
    return slice.reduce((a, b) => a + b, 0) / period
  })
}

function CandlestickChart({ data }: { data: any[] }) {
  if (!data.length) return null
  const recent = data.slice(-90)
  const allVals = recent.flatMap((d: any) => [d.high, d.low])
  const yMin = Math.min(...allVals) * 0.998
  const yMax = Math.max(...allVals) * 1.002
  const barWidth = Math.max(1, Math.floor(600 / recent.length) - 1)

  return (
    <div className="overflow-x-auto">
      <svg width="100%" viewBox={`0 0 ${recent.length * (barWidth + 2) + 40} 220`} className="font-mono">
        <defs>
          <linearGradient id="volGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="hsl(var(--cyan))" stopOpacity="0.3" />
            <stop offset="100%" stopColor="hsl(var(--cyan))" stopOpacity="0.05" />
          </linearGradient>
        </defs>
        {recent.map((d: any, i: number) => {
          const x = i * (barWidth + 2) + 20
          const green = d.close >= d.open
          const color = green ? 'hsl(152, 70%, 45%)' : 'hsl(350, 70%, 55%)'
          const yScale = (v: number) => 20 + ((yMax - v) / (yMax - yMin)) * 180
          const bodyTop = yScale(Math.max(d.open, d.close))
          const bodyBot = yScale(Math.min(d.open, d.close))
          const bodyH = Math.max(bodyBot - bodyTop, 1)
          return (
            <g key={i}>
              <line x1={x + barWidth / 2} y1={yScale(d.high)} x2={x + barWidth / 2} y2={yScale(d.low)} stroke={color} strokeWidth={1} />
              <rect x={x} y={bodyTop} width={barWidth} height={bodyH} fill={color} rx={0.5} />
            </g>
          )
        })}
        <text x="5" y="210" fontSize="8" fill="hsl(var(--muted-foreground))">90d</text>
      </svg>
    </div>
  )
}

export default function Predictions() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [period, setPeriod] = useState('2y')
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<any>(null)
  const debouncedTicker = useDebouncedValue(ticker, 400)

  const { data, loading, error, refetch } = useApi<any>(`/api/predictions/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const { data: featureImp } = useApi<any>(`/api/predictions/${debouncedTicker}/feature-importance`)
  const { data: pipeStatus } = useApi<any>('/api/pipeline/status')

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

  const longTermChart = useMemo(() => {
    if (!chartData?.length) return []
    const closes = chartData.map((d: any) => d.close)
    const sma50 = computeSMA(closes, 50)
    const sma200 = computeSMA(closes, 200)
    return chartData.map((d: any, i: number) => ({
      date: d.date?.slice(0, 10),
      close: d.close,
      sma50: sma50[i] ? Math.round(sma50[i]! * 100) / 100 : null,
      sma200: sma200[i] ? Math.round(sma200[i]! * 100) / 100 : null,
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

  const handleTrain = useCallback(async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const res = await fetch(`/api/pipeline/train/${ticker}`, { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') {
        const poll = setInterval(async () => {
          const statusRes = await fetch(`/api/pipeline/train/${ticker}/status`)
          const status = await statusRes.json()
          if (status.status === 'done') {
            setTrainResult(status.result)
            setTraining(false)
            clearInterval(poll)
            refetch()
          } else if (status.status === 'error') {
            setTrainResult({ error: status.error })
            setTraining(false)
            clearInterval(poll)
          }
        }, 2000)
      } else {
        setTraining(false)
      }
    } catch {
      setTraining(false)
    }
  }, [ticker, refetch])

  const now = new Date()
  const isWeekend = now.getDay() === 0 || now.getDay() === 6
  const hour = now.getHours()
  const min = now.getMinutes()
  const marketOpen = hour * 60 + min >= 555 && hour * 60 + min <= 930
  const marketStatus = isWeekend ? 'Closed (Weekend)' : marketOpen ? 'Open' : 'Closed'
  const isMarketOpen = marketStatus === 'Open'

  return (
    <div className="space-y-6">
      <PageHeader
        title="Predictions"
        description="5-model ML ensemble signals"
        badge="Live"
      >
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-full border border-border bg-card/70 px-3 py-1.5 text-xs">
            <span className={`h-2 w-2 rounded-full ${isMarketOpen ? 'bg-emerald animate-pulse' : 'bg-rose'}`} />
            <span className="text-muted-foreground">{marketStatus}</span>
          </div>
          <span className="rounded-full border border-border bg-card/70 px-3 py-1.5 text-xs font-mono text-muted-foreground">{now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}</span>
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
          >
            {stocks.data?.stocks?.map((s: string) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <button
            onClick={refetch}
            className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono hover:bg-accent/50 transition-colors"
          >
            Refresh
          </button>
          <button
            onClick={handleTrain}
            disabled={training}
            className="rounded-lg bg-cyan px-4 py-2 text-sm font-mono font-semibold text-black transition-colors hover:opacity-90 disabled:opacity-50"
          >
            {training ? 'Training...' : 'Train Model'}
          </button>
        </div>
      </PageHeader>

      {training && (
        <Card className="border-cyan/30 bg-cyan/5">
          <div className="flex items-center gap-3">
            <Spinner />
            <div>
              <p className="text-sm font-semibold">Training model for {ticker}</p>
              <p className="text-xs text-muted-foreground">This may take 1-3 minutes. Fetching 5y data, training LSTM, GRU, Transformer, XGBoost, LightGBM...</p>
            </div>
          </div>
        </Card>
      )}

      {trainResult && !training && (
        <Card className={trainResult.error ? 'border-rose/30 bg-rose/5' : 'border-emerald/30 bg-emerald/5'}>
          {trainResult.error ? (
            <p className="text-sm text-rose">Training failed: {trainResult.error}</p>
          ) : (
            <div className="grid grid-cols-3 md:grid-cols-6 gap-3 text-center">
              <div><p className="text-xs text-muted-foreground">XGBoost</p><p className="font-mono text-sm font-bold">{(trainResult.xgb_accuracy * 100).toFixed(1)}%</p></div>
              <div><p className="text-xs text-muted-foreground">LightGBM</p><p className="font-mono text-sm font-bold">{(trainResult.lgb_accuracy * 100).toFixed(1)}%</p></div>
              <div><p className="text-xs text-muted-foreground">LSTM</p><p className="font-mono text-sm font-bold">{(trainResult.lstm_accuracy * 100).toFixed(1)}%</p></div>
              <div><p className="text-xs text-muted-foreground">GRU</p><p className="font-mono text-sm font-bold">{(trainResult.gru_accuracy * 100).toFixed(1)}%</p></div>
              <div><p className="text-xs text-muted-foreground">Transformer</p><p className="font-mono text-sm font-bold">{(trainResult.transformer_accuracy * 100).toFixed(1)}%</p></div>
              <div><p className="text-xs text-muted-foreground">Ensemble</p><p className="font-mono text-sm font-bold text-cyan">{(trainResult.ensemble_accuracy * 100).toFixed(1)}%</p></div>
            </div>
          )}
        </Card>
      )}

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          {/* Hero Prediction */}
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
              </>
            ) : (
              <EmptyState message="Models not trained for this ticker. Click 'Train Model' to start." />
            )}
          </Card>

          {/* Metrics Row */}
          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
              <Stat
                label="Price"
                value={`₹${metrics.price}`}
                sub={`${metrics.day_change >= 0 ? '+' : ''}${metrics.day_change} (${metrics.day_change_pct}%)`}
                trend={metrics.day_change >= 0 ? 'up' : 'down'}
              />
              <Stat label="RSI (14)" value={metrics.rsi.toFixed(1)} sub={metrics.rsi > 70 ? 'Overbought' : metrics.rsi < 30 ? 'Oversold' : 'Neutral'} trend={metrics.rsi > 70 ? 'down' : metrics.rsi < 30 ? 'up' : 'neutral'} />
              <Stat label="MACD" value={metrics.macd.toFixed(2)} sub={metrics.macd > 0 ? 'Bullish' : 'Bearish'} trend={metrics.macd > 0 ? 'up' : 'down'} />
              <Stat label="ATR" value={metrics.atr.toFixed(2)} sub="Avg True Range" />
              <Stat label="Volume" value={`${(metrics.volume / 1_000_000).toFixed(1)}M`} />
              {pred?.details && (
                <>
                  <Stat label="LSTM" value={`${(pred.details.lstm_prob * 100).toFixed(1)}%`} sub={pred.details.lstm_dir === 1 ? '▲ UP' : '▼ DN'} trend={pred.details.lstm_dir === 1 ? 'up' : 'down'} />
                  <Stat label="Transformer" value={`${(pred.details.transformer_prob * 100).toFixed(1)}%`} sub={pred.details.transformer_dir === 1 ? '▲ UP' : '▼ DN'} trend={pred.details.transformer_dir === 1 ? 'up' : 'down'} />
                </>
              )}
            </div>
          )}

          {/* Candlestick Chart */}
          {chartData?.length > 0 && (
            <>
              <SectionHeader title="Price Action (90d Candlestick)" />
              <Card className="overflow-hidden">
                <CandlestickChart data={chartData} />
              </Card>
            </>
          )}

          {/* Price Line + SMA + Volume Chart */}
          {enrichedChart.length > 0 && (
            <>
              <SectionHeader title="Price & Moving Averages" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={300}>
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
                  </ComposedChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {/* RSI Subplot */}
          {enrichedChart.length > 0 && (
            <>
              <SectionHeader title="RSI (14)" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={140}>
                  <ComposedChart data={enrichedChart} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} interval="preserveStartEnd" />
                    <YAxis domain={[0, 100]} ticks={[30, 50, 70]} tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} width={30} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                    <ReferenceLine y={70} stroke="#ef4444" strokeDasharray="3 3" strokeWidth={1} />
                    <ReferenceLine y={30} stroke="#22c55e" strokeDasharray="3 3" strokeWidth={1} />
                    <Area type="monotone" dataKey="rsi" stroke="#06b6d4" fill="#06b6d4" fillOpacity={0.15} strokeWidth={1.5} dot={false} name="RSI" />
                  </ComposedChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {/* Volume Chart */}
          {enrichedChart.length > 0 && (
            <>
              <SectionHeader title="Volume" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={120}>
                  <BarChart data={enrichedChart} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} width={35} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} formatter={(v: number) => [`${v.toFixed(1)}M`, 'Volume']} />
                    <Bar dataKey="vol" fill="hsl(var(--cyan) / 0.5)" radius={[1, 1, 0, 0]} name="Volume (M)" />
                  </BarChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {/* Long-term Trend */}
          {longTermChart.length > 0 && (
            <>
              <SectionHeader title="Long Term Trend (SMA 50 / SMA 200)" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={280}>
                  <LineChart data={longTermChart} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="date" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))' }} interval="preserveStartEnd" />
                    <YAxis domain={['auto', 'auto']} tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={55} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} />
                    <Line type="monotone" dataKey="close" stroke="hsl(var(--foreground))" dot={false} strokeWidth={1.5} name="Close" />
                    <Line type="monotone" dataKey="sma50" stroke="#8b5cf6" dot={false} strokeWidth={1} strokeDasharray="4 4" name="SMA 50" connectNulls />
                    <Line type="monotone" dataKey="sma200" stroke="#f59e0b" dot={false} strokeWidth={1.5} strokeDasharray="6 3" name="SMA 200" connectNulls />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {/* Feature Importance + Model Votes */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {featureImp?.features && (
              <div>
                <SectionHeader title="Feature Importance" />
                <Card>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={featureImp.features.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 10 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                      <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={100} />
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
            )}

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
                          <span className={`w-10 text-xs font-mono font-bold ${m.dir === 1 ? 'text-emerald' : 'text-rose'}`}>
                            {m.dir === 1 ? '▲' : '▼'} {dirLabel}
                          </span>
                          <div className="flex-1 h-2.5 rounded-full bg-muted overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${m.dir === 1 ? 'bg-emerald' : 'bg-rose'}`}
                              style={{ width: `${(m.prob || 0) * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs w-14 text-right">{((m.prob || 0) * 100).toFixed(1)}%</span>
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
                        <td className="py-2.5 font-mono text-xs">{row.date?.slice(5, 10)}</td>
                        <td className="py-2.5 text-right font-mono">{row.open}</td>
                        <td className="py-2.5 text-right font-mono">{row.high}</td>
                        <td className="py-2.5 text-right font-mono">{row.low}</td>
                        <td className="py-2.5 text-right font-mono font-semibold">{row.close}</td>
                        <td className={`py-2.5 text-right font-mono ${row.chgPct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {row.chgPct >= 0 ? '+' : ''}{row.chgPct}%
                        </td>
                        <td className="py-2.5 text-right font-mono text-muted-foreground">{(row.volume / 1_000_000).toFixed(1)}M</td>
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

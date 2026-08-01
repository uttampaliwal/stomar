import { useState, useMemo, useCallback } from 'react'
import {
  ComposedChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, Cell, Area, BarChart
} from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { apiFetch } from '@/lib/api-client'
import TradingViewChart, { type TradeMarker, type HorizontalLine } from '@/components/TradingViewChart'
import { OrderPad } from '@/components/TerminalComponents'
import type { PredictionResponse, FeatureImportanceResponse, TrainResponse } from '../lib/api-types'

function computeSMA(data: number[], period: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < period - 1) return null
    const slice = data.slice(i - period + 1, i + 1)
    return slice.reduce((a, b) => a + b, 0) / period
  })
}

export default function Predictions() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const [training, setTraining] = useState(false)
  const [trainResult, setTrainResult] = useState<TrainResponse | null>(null)
  const [showOrderPad, setShowOrderPad] = useState(false)
  const debouncedTicker = useDebouncedValue(ticker, 400)

  const { data, loading, error, refetch } = useApi<PredictionResponse>(`/api/predictions/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const { data: featureImp } = useApi<FeatureImportanceResponse>(`/api/predictions/${debouncedTicker}/feature-importance`)

  const pred = data?.prediction
  const metrics = data?.metrics
  const chartData = data?.chart
  const recentData = data?.recent

  const enrichedChart = useMemo(() => {
    if (!chartData?.length) return []
    const closes = chartData.map((d) => d.close)
    const sma20 = computeSMA(closes, 20)
    const sma50 = computeSMA(closes, 50)
    return chartData.map((d, i) => ({
      ...d,
      date: d.date?.slice(5, 10),
      sma20: sma20[i] ? Math.round(sma20[i]! * 100) / 100 : null,
      sma50: sma50[i] ? Math.round(sma50[i]! * 100) / 100 : null,
      vol: d.volume / 1_000_000,
    }))
  }, [chartData])

  const enrichedRecent = useMemo(() => {
    if (!recentData?.length) return []
    return [...recentData].reverse().map((r, i, arr) => {
      const prevClose = i > 0 ? arr[i - 1].close : r.open
      const chgPct = prevClose ? ((r.close - prevClose) / prevClose * 100) : 0
      return { ...r, chgPct: Math.round(chgPct * 100) / 100 }
    })
  }, [recentData])

  // Build trade markers from prediction
  const tradeMarkers = useMemo((): TradeMarker[] => {
    if (!chartData?.length || !pred) return []
    const lastCandle = chartData[chartData.length - 1]
    if (!lastCandle) return []
    return [{
      time: lastCandle.date,
      position: pred.direction === 'BUY' ? 'belowBar' as const : 'aboveBar' as const,
      color: pred.direction === 'BUY' ? '#10b981' : '#f43f5e',
      shape: pred.direction === 'BUY' ? 'arrowUp' as const : 'arrowDown' as const,
      text: `${pred.direction} ${(pred.confidence * 100).toFixed(0)}%`,
    }]
  }, [chartData, pred])

  // Build horizontal lines for SL/TP
  const hLines = useMemo((): HorizontalLine[] => {
    if (!metrics) return []
    const lines: HorizontalLine[] = []
    // ATR-based stop loss and take profit
    const atr = metrics.atr || 0
    const price = metrics.price || 0
    if (atr > 0 && price > 0) {
      lines.push({
        price: price - 2 * atr,
        color: '#f43f5e',
        lineStyle: 2, // Dashed
        title: 'SL (2x ATR)',
        axisLabelVisible: true,
      })
      lines.push({
        price: price + 3 * atr,
        color: '#10b981',
        lineStyle: 2,
        title: 'TP (3x ATR)',
        axisLabelVisible: true,
      })
    }
    return lines
  }, [metrics])

  const handleTrain = useCallback(async () => {
    setTraining(true)
    setTrainResult(null)
    try {
      const res = await apiFetch(`/api/pipeline/train/${ticker}`, { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') {
        const poll = setInterval(async () => {
          const statusRes = await apiFetch(`/api/pipeline/train/${ticker}/status`)
          const status = await statusRes.json()
          if (status.status === 'done') {
            setTrainResult(status.result)
            setTraining(false)
            clearInterval(poll)
            refetch()
          } else if (status.status === 'error') {
            setTrainResult({ error: status.error } as TrainResponse)
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
    <div className="space-y-4 grid-lines min-h-screen">
      <PageHeader
        title="Predictions"
        description="5-model ML ensemble signals with TradingView charts"
        badge="Live"
      >
        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5 rounded-full border border-border bg-card/70 px-3 py-1.5 text-xs">
            <span className={`h-2 w-2 rounded-full ${isMarketOpen ? 'bg-emerald animate-pulse' : 'bg-rose'}`} />
            <span className="text-muted-foreground">{marketStatus}</span>
          </div>
          <span className="rounded-full border border-border bg-card/70 px-3 py-1.5 text-xs font-mono text-muted-foreground">
            {now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' })}
          </span>
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
            onClick={() => setShowOrderPad(!showOrderPad)}
            className="rounded-lg border border-amber/30 bg-amber/10 px-3 py-2 text-sm font-mono text-amber hover:bg-amber/20 transition-colors"
          >
            Order Pad
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
        <Card className="border-cyan/30 bg-cyan/5 !p-3">
          <div className="flex items-center gap-3">
            <Spinner size="sm" />
            <div>
              <p className="text-sm font-semibold">Training model for {ticker}</p>
              <p className="text-xs text-muted-foreground">LSTM, GRU, Transformer, XGBoost, LightGBM ensemble...</p>
            </div>
          </div>
        </Card>
      )}

      {trainResult && !training && (
        <Card className={trainResult.error ? 'border-rose/30 bg-rose/5 !p-3' : 'border-emerald/30 bg-emerald/5 !p-3'}>
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

      {data && !('error' in data) && (
        <div className="grid gap-4 lg:grid-cols-[1fr_280px]">
          {/* Main Chart Column */}
          <div className="space-y-4">
            {/* Hero Prediction */}
            <Card className="text-center py-4 !p-4">
              {pred ? (
                <div className="flex items-center justify-center gap-4">
                  <Badge
                    variant={pred.direction === 'BUY' ? 'success' : 'danger'}
                    className="text-lg px-5 py-1"
                  >
                    {pred.direction === 'BUY' ? '▲' : '▼'} {pred.direction}
                  </Badge>
                  <div>
                    <p className="font-mono text-3xl font-bold">
                      {((pred.confidence || 0) * 100).toFixed(1)}%
                    </p>
                    <p className="text-xs text-muted-foreground">Ensemble Confidence</p>
                  </div>
                  {pred?.details && (
                    <div className="flex gap-3 text-xs font-mono">
                      <span className="text-emerald">▲ {((pred.details.lstm_prob as number) * 100).toFixed(0)}% LSTM</span>
                      <span className="text-emerald">▲ {((pred.details.transformer_prob as number) * 100).toFixed(0)}% XFR</span>
                    </div>
                  )}
                </div>
              ) : (
                <EmptyState message="Models not trained for this ticker." />
              )}
            </Card>

            {/* Metrics Row */}
            {metrics && (
              <div className="grid grid-cols-3 md:grid-cols-4 lg:grid-cols-7 gap-2">
                <Stat label="Price" value={`₹${metrics.price}`} sub={`${metrics.day_change >= 0 ? '+' : ''}${metrics.day_change_pct}%`} trend={metrics.day_change >= 0 ? 'up' : 'down'} className="!p-2" />
                <Stat label="RSI" value={metrics.rsi.toFixed(1)} sub={metrics.rsi > 70 ? 'Overbought' : metrics.rsi < 30 ? 'Oversold' : 'Neutral'} trend={metrics.rsi > 70 ? 'down' : metrics.rsi < 30 ? 'up' : 'neutral'} className="!p-2" />
                <Stat label="MACD" value={metrics.macd.toFixed(2)} sub={metrics.macd > 0 ? 'Bullish' : 'Bearish'} trend={metrics.macd > 0 ? 'up' : 'down'} className="!p-2" />
                <Stat label="ATR" value={metrics.atr.toFixed(2)} className="!p-2" />
                <Stat label="Volume" value={`${(metrics.volume / 1_000_000).toFixed(1)}M`} className="!p-2" />
                {pred?.details && (
                  <>
                    <Stat label="LSTM" value={`${((pred.details.lstm_prob as number) * 100).toFixed(1)}%`} sub={pred.details.lstm_dir === 1 ? '▲ UP' : '▼ DN'} trend={pred.details.lstm_dir === 1 ? 'up' : 'down'} className="!p-2" />
                    <Stat label="XFR" value={`${((pred.details.transformer_prob as number) * 100).toFixed(1)}%`} sub={pred.details.transformer_dir === 1 ? '▲ UP' : '▼ DN'} trend={pred.details.transformer_dir === 1 ? 'up' : 'down'} className="!p-2" />
                  </>
                )}
              </div>
            )}

            {/* TradingView Chart */}
            {chartData && chartData.length > 0 && (
              <>
                <SectionHeader title="Price Action" />
                <Card className="!p-0 overflow-hidden">
                  <TradingViewChart
                    data={chartData.slice(-120)}
                    height={420}
                    sma20
                    sma50
                    bollingerBands
                    volume
                    markers={tradeMarkers}
                    horizontalLines={hLines}
                  />
                </Card>
              </>
            )}

            {/* RSI Chart */}
            {enrichedChart.length > 0 && (
              <>
                <SectionHeader title="RSI (14)" />
                <Card className="!p-0 overflow-hidden">
                  <ResponsiveContainer width="100%" height={120}>
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

            {/* Feature Importance + Model Votes */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
              {featureImp?.features && (
                <div>
                  <SectionHeader title="Feature Importance" />
                  <Card className="!p-3">
                    <ResponsiveContainer width="100%" height={260}>
                      <BarChart data={featureImp.features.slice(0, 8)} layout="vertical" margin={{ left: 10, right: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                        <XAxis type="number" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                        <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={100} />
                        <Tooltip
                          contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '12px' }}
                          formatter={(v: number) => [(v * 100).toFixed(2) + '%', 'Importance']}
                        />
                        <Bar dataKey="importance" radius={[0, 4, 4, 0]}>
                          {featureImp.features.slice(0, 8).map((f, i) => {
                            const avg = featureImp.features.reduce((a, b) => a + b.importance, 0) / featureImp.features.length
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
                  <Card className="!p-3">
                    <div className="space-y-2.5">
                      {[
                        { name: 'LSTM', dir: pred.details.lstm_dir, prob: pred.details.lstm_prob },
                        { name: 'GRU', dir: pred.details.gru_dir, prob: pred.details.gru_prob },
                        { name: 'Transformer', dir: pred.details.transformer_dir, prob: pred.details.transformer_prob },
                        { name: 'XGBoost', dir: pred.details.xgb_dir, prob: pred.details.xgb_prob_up },
                        { name: 'LightGBM', dir: pred.details.lgb_dir, prob: pred.details.lgb_prob_up },
                      ].map((m) => (
                        <div key={m.name} className="flex items-center gap-2">
                          <span className="w-20 text-[0.65rem] font-mono uppercase text-muted-foreground">{m.name}</span>
                          <span className={`w-8 text-[0.65rem] font-mono font-bold ${m.dir === 1 ? 'text-emerald' : 'text-rose'}`}>
                            {m.dir === 1 ? '▲' : '▼'}
                          </span>
                          <div className="flex-1 h-2 rounded-full bg-muted/30 overflow-hidden">
                            <div
                              className={`h-full rounded-full transition-all ${m.dir === 1 ? 'bg-emerald' : 'bg-rose'}`}
                              style={{ width: `${((m.prob as number) || 0) * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-[0.65rem] w-12 text-right">{(((m.prob as number) || 0) * 100).toFixed(1)}%</span>
                        </div>
                      ))}
                    </div>
                  </Card>
                </div>
              )}
            </div>

            {/* Recent Prices Table */}
            {enrichedRecent.length > 0 && (
              <>
                <SectionHeader title="Recent Prices" />
                <Card className="overflow-x-auto !p-0">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-border text-[0.6rem] text-muted-foreground uppercase font-mono">
                        <th className="text-left py-2 px-3">Date</th>
                        <th className="text-right py-2 px-3">Open</th>
                        <th className="text-right py-2 px-3">High</th>
                        <th className="text-right py-2 px-3">Low</th>
                        <th className="text-right py-2 px-3">Close</th>
                        <th className="text-right py-2 px-3">Chg%</th>
                        <th className="text-right py-2 px-3">Volume</th>
                      </tr>
                    </thead>
                    <tbody>
                      {enrichedRecent.map((row, i) => (
                        <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                          <td className="py-2 px-3 font-mono text-xs">{row.date?.slice(5, 10)}</td>
                          <td className="py-2 px-3 text-right font-mono text-xs">{row.open}</td>
                          <td className="py-2 px-3 text-right font-mono text-xs">{row.high}</td>
                          <td className="py-2 px-3 text-right font-mono text-xs">{row.low}</td>
                          <td className="py-2 px-3 text-right font-mono text-xs font-semibold">{row.close}</td>
                          <td className={`py-2 px-3 text-right font-mono text-xs ${row.chgPct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                            {row.chgPct >= 0 ? '+' : ''}{row.chgPct}%
                          </td>
                          <td className="py-2 px-3 text-right font-mono text-xs text-muted-foreground">{(row.volume / 1_000_000).toFixed(1)}M</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </Card>
              </>
            )}
          </div>

          {/* Right Sidebar: Order Pad */}
          {showOrderPad && metrics && (
            <div className="space-y-4 animate-slide-in-right">
              <OrderPad
                ticker={ticker}
                currentPrice={metrics.price}
                onClose={() => setShowOrderPad(false)}
              />
            </div>
          )}
        </div>
      )}
    </div>
  )
}

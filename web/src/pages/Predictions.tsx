import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatPercent } from '@/lib/utils'

export default function Predictions() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const { data, loading, error } = useApi<any>(`/api/predictions/${ticker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const pred = data?.prediction
  const metrics = data?.metrics

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Predictions</h1>
          <p className="text-sm text-muted-foreground">5-model ML ensemble signals</p>
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
          {/* Prediction Hero */}
          <Card className="text-center py-6">
            {pred ? (
              <>
                <Badge variant={pred.direction === 'BUY' ? 'success' : 'danger'} className="text-lg px-4 py-1 mb-2">
                  {pred.direction}
                </Badge>
                <p className="font-mono text-3xl font-bold mt-2">{(pred.confidence * 100).toFixed(1)}%</p>
                <p className="text-sm text-muted-foreground mt-1">Confidence</p>
              </>
            ) : (
              <EmptyState message="Models not trained for this ticker" />
            )}
          </Card>

          {/* Metrics */}
          {metrics && (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
              <Stat label="Price" value={`₹${metrics.price}`} trend={metrics.day_change >= 0 ? 'up' : 'down'} />
              <Stat label="Day Change" value={formatPercent(metrics.day_change_pct)} trend={metrics.day_change_pct >= 0 ? 'up' : 'down'} />
              <Stat label="RSI" value={metrics.rsi.toFixed(1)} sub={metrics.rsi > 70 ? 'Overbought' : metrics.rsi < 30 ? 'Oversold' : 'Neutral'} />
              <Stat label="Volume" value={metrics.volume.toLocaleString()} />
            </div>
          )}

          {/* Model Breakdown */}
          {pred?.details && (
            <>
              <SectionHeader title="Model Votes" />
              <Card>
                <div className="space-y-2">
                  {Object.entries(pred.details).filter(([k]) => !k.includes('weight') && !k.includes('pred')).map(([model, confidence]) => (
                    <div key={model} className="flex items-center gap-3">
                      <span className="w-24 text-xs font-mono uppercase text-muted-foreground">{model}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${Number(confidence) > 0.5 ? 'bg-emerald' : 'bg-rose'}`}
                          style={{ width: `${Number(confidence) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs w-12 text-right">{(Number(confidence) * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          {/* Recent Prices */}
          {data.recent && (
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
                      <th className="text-right py-2">Volume</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.recent.map((row: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                        <td className="py-2 font-mono text-xs">{row.date}</td>
                        <td className="py-2 text-right font-mono">{row.open}</td>
                        <td className="py-2 text-right font-mono">{row.high}</td>
                        <td className="py-2 text-right font-mono">{row.low}</td>
                        <td className="py-2 text-right font-mono font-semibold">{row.close}</td>
                        <td className="py-2 text-right font-mono text-muted-foreground">{row.volume.toLocaleString()}</td>
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

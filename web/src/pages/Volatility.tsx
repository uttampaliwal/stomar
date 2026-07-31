import { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'
import { formatPercent } from '@/lib/utils'
import type { VolatilityResponse } from '../lib/api-types'

export default function Volatility() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<VolatilityResponse>(`/api/volatility/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const forecastData = data?.forecast?.forecast_vols?.map((v: number, i: number) => ({
    day: `Day ${i + 1}`,
    vol: Math.round(v * 10000) / 100,
    current: data?.forecast?.current_vol ? Math.round(data.forecast.current_vol * 10000) / 100 : null,
    longTerm: data?.forecast?.long_term_vol ? Math.round(data.forecast.long_term_vol * 10000) / 100 : null,
  })) || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Volatility Analysis"
        description="Multiple volatility estimators, regime detection, and forecasting"
        badge="Risk"
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

      {data && !('error' in data) ? (
        <>
          {data.regime && (
            <Card className="text-center py-4">
              <Badge variant={data.regime === 'High' ? 'danger' : data.regime === 'Low' ? 'success' : 'warning'} className="text-lg px-4 py-1">
                {data.regime} Volatility
              </Badge>
              {data.percentile !== undefined && (
                <p className="text-sm text-muted-foreground mt-2">Percentile: {data.percentile.toFixed(0)}%</p>
              )}
            </Card>
          )}

          <SectionHeader title="Volatility Metrics" />
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-5 gap-3">
            <Stat label="Historical" value={formatPercent(data.historical_vol * 100)} />
            <Stat label="EWMA" value={formatPercent(data.ewma_vol * 100)} />
            <Stat label="Parkinson" value={formatPercent(data.parkinson_vol * 100)} />
            <Stat label="Garman-Klass" value={formatPercent(data.garman_klass_vol * 100)} />
            <Stat label="Yang-Zhang" value={formatPercent(data.yang_zhang_vol * 100)} />
            <Stat label="Current Vol" value={formatPercent(data.current_vol * 100)} />
            <Stat label="ATR %" value={formatPercent(data.atr_pct * 100)} />
            <Stat label="BB Width" value={(data.bb_width || 0).toFixed(4)} />
            <Stat label="BB %B" value={(data.bb_pct_b || 0).toFixed(4)} />
            <Stat label="Percentile" value={`${(data.percentile || 0).toFixed(0)}%`} />
          </div>

          {/* Forecast Chart */}
          {forecastData.length > 0 && (
            <>
              <SectionHeader title="5-Day Volatility Forecast" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={200}>
                  <LineChart data={forecastData} margin={{ top: 5, right: 5, bottom: 0, left: 0 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="day" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} width={40} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} formatter={(v: number) => [`${v.toFixed(2)}%`]} />
                    <ReferenceLine y={data.forecast?.current_vol ? data.forecast.current_vol * 100 : 0} stroke="#06b6d4" strokeDasharray="4 4" label={{ value: 'Current', position: 'right', fontSize: 9 }} />
                    <ReferenceLine y={data.forecast?.long_term_vol ? data.forecast.long_term_vol * 100 : 0} stroke="#f59e0b" strokeDasharray="4 4" label={{ value: 'Long-term', position: 'right', fontSize: 9 }} />
                    <Line type="monotone" dataKey="vol" stroke="#06b6d4" dot={{ r: 3 }} strokeWidth={2} name="Forecast Vol %" />
                  </LineChart>
                </ResponsiveContainer>
              </Card>
            </>
          )}

          {data.position_sizing && (
            <>
              <SectionHeader title="Position Sizing" />
              <Card>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Recommended Size</p>
                    <p className="font-mono text-xl font-bold text-cyan">{((data.position_sizing.recommended_size || 0) * 100).toFixed(1)}%</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Vol Scalar</p>
                    <p className="font-mono text-xl font-bold">{(data.position_sizing.vol_scalar || 0).toFixed(2)}</p>
                  </div>
                  <div>
                    <p className="text-xs text-muted-foreground uppercase">Reasoning</p>
                    <p className="text-sm text-muted-foreground">{data.position_sizing.reasoning}</p>
                  </div>
                </div>
              </Card>
            </>
          )}
        </>
      ) : (
        !loading && <EmptyState message="No volatility data is available for the selected ticker." />
      )}
    </div>
  )
}

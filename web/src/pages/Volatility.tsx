import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatPercent } from '@/lib/utils'

export default function Volatility() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const { data, loading, error } = useApi<any>(`/api/volatility/${ticker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Volatility</h1>
          <p className="text-sm text-muted-foreground">GARCH, Parkinson, Garman-Klass, Yang-Zhang</p>
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
          {data.regime && (
            <Card className="text-center py-4">
              <Badge variant={data.regime === 'High' ? 'danger' : data.regime === 'Low' ? 'success' : 'warning'} className="text-lg px-4 py-1">
                {data.regime} Volatility
              </Badge>
            </Card>
          )}

          <SectionHeader title="Volatility Metrics" />
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {data.historical_vol !== undefined && <Stat label="Historical" value={formatPercent(data.historical_vol * 100)} />}
            {data.ewma_vol !== undefined && <Stat label="EWMA" value={formatPercent(data.ewma_vol * 100)} />}
            {data.parkinson_vol !== undefined && <Stat label="Parkinson" value={formatPercent(data.parkinson_vol * 100)} />}
            {data.garman_klass_vol !== undefined && <Stat label="Garman-Klass" value={formatPercent(data.garman_klass_vol * 100)} />}
            {data.yang_zhang_vol !== undefined && <Stat label="Yang-Zhang" value={formatPercent(data.yang_zhang_vol * 100)} />}
            {data.current_vol !== undefined && <Stat label="Current Vol" value={formatPercent(data.current_vol * 100)} />}
            {data.percentile !== undefined && <Stat label="Percentile" value={`${data.percentile.toFixed(0)}%`} />}
            {data.atr_pct !== undefined && <Stat label="ATR %" value={formatPercent(data.atr_pct * 100)} />}
          </div>

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
                    <p className="text-sm text-muted-foreground">{data.position_sizing.reasoning || '—'}</p>
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

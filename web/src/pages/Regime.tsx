import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

export default function Regime() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/regime/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Regime Detection</h1>
          <p className="text-sm text-muted-foreground">Bull / Bear / Sideways market regime</p>
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
          <Card className="text-center py-6">
            <Badge variant={data.regime === 'Bull' ? 'success' : data.regime === 'Bear' ? 'danger' : 'warning'} className="text-xl px-6 py-1.5">
              {data.regime || 'Unknown'}
            </Badge>
            <p className="text-sm text-muted-foreground mt-3">Confidence: {data.confidence > 1 ? (data.confidence || 0).toFixed(1) : ((data.confidence || 0) * 100).toFixed(1)}%</p>
          </Card>

          {data.recommendation && (
            <>
              <SectionHeader title="Recommendation" />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <Card>
                  <p className="text-xs text-muted-foreground uppercase">Action</p>
                  <p className="font-mono text-lg font-bold mt-1">{data.recommendation.action}</p>
                </Card>
                <Card>
                  <p className="text-xs text-muted-foreground uppercase">Allocation</p>
                  <p className="font-mono text-lg font-bold mt-1">{data.recommendation.allocation}</p>
                </Card>
                <Card>
                  <p className="text-xs text-muted-foreground uppercase">Risk Level</p>
                  <p className="font-mono text-lg font-bold mt-1">{data.recommendation.risk_level}</p>
                </Card>
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

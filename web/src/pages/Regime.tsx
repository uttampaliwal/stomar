import { useState } from 'react'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

export default function Regime() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/regime/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const regime = data?.regime
  const indicators = data?.indicators || {}
  const recommendation = data?.recommendation
  const performance = data?.performance

  const regimeColor = regime === 'Bull' ? 'text-emerald' : regime === 'Bear' ? 'text-rose' : 'text-muted-foreground'
  const regimeBadge = regime === 'Bull' ? 'success' : regime === 'Bear' ? 'danger' : 'warning'

  return (
    <div className="space-y-6">
      <PageHeader
        title="Market Regime"
        description="Current market environment detection"
        badge="Signal"
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
          {regime && (
            <Card className="text-center py-6">
              <Badge variant={regimeBadge} className="text-3xl px-6 py-2 font-mono">
                {regime === 'Bull' ? '▲ BULL' : regime === 'Bear' ? '▼ BEAR' : '● NEUTRAL'}
              </Badge>
              {data.confidence !== undefined && (
                <p className="text-sm text-muted-foreground mt-3">
                  Confidence: {data.confidence > 1 ? `${data.confidence.toFixed(0)}%` : `${(data.confidence * 100).toFixed(0)}%`}
                </p>
              )}
            </Card>
          )}

          {recommendation && (
            <Card>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-center">
                <div>
                  <p className="text-xs text-muted-foreground uppercase">Recommended Action</p>
                  <p className="font-mono text-xl font-bold text-cyan">{recommendation.action}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase">Allocation</p>
                  <p className="font-mono text-sm font-bold">{recommendation.allocation}</p>
                </div>
                <div>
                  <p className="text-xs text-muted-foreground uppercase">Risk Level</p>
                  <Badge variant={recommendation.risk_level === 'LOW' ? 'success' : recommendation.risk_level === 'HIGH' ? 'danger' : 'warning'}>
                    {recommendation.risk_level}
                  </Badge>
                </div>
              </div>
            </Card>
          )}

          {data.bull_signals !== undefined && (
            <Card>
              <div className="flex items-center justify-center gap-6">
                <div className="text-center">
                  <p className="text-xs text-muted-foreground uppercase">Bull Signals</p>
                  <p className="font-mono text-2xl font-bold text-emerald">{data.bull_signals || 0}</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-muted-foreground uppercase">Bear Signals</p>
                  <p className="font-mono text-2xl font-bold text-rose">{data.bear_signals || 0}</p>
                </div>
                <div className="text-center">
                  <p className="text-xs text-muted-foreground uppercase">Total Signals</p>
                  <p className="font-mono text-2xl font-bold text-muted-foreground">{data.total_signals || 0}</p>
                </div>
              </div>
            </Card>
          )}

          {Object.keys(indicators).length > 0 && (
            <>
              <SectionHeader title="Technical Indicators" />
              <Card>
                <div className="space-y-2">
                  {Object.entries(indicators).map(([key, value]) => (
                    <div key={key} className="flex justify-between items-center py-1 border-b border-border/50">
                      <span className="text-sm text-muted-foreground">{key}</span>
                      <Badge variant={
                        (value as string)?.toLowerCase().includes('bullish') || (value as string)?.toLowerCase().includes('strong') ? 'success' :
                        (value as string)?.toLowerCase().includes('bearish') || (value as string)?.toLowerCase().includes('weak') ? 'danger' :
                        'default'
                      }>
                        {value as string}
                      </Badge>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          {performance && (
            <>
              <SectionHeader title="Regime Performance" />
              <Card>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  {performance.returns && typeof performance.returns === 'object' && Object.entries(performance.returns).map(([key, value]) => (
                    <div key={key}>
                      <p className="text-xs text-muted-foreground uppercase">{key}</p>
                      <p className={`font-mono text-lg font-bold ${(value as number) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                        {((value as number) * 100).toFixed(2)}%
                      </p>
                    </div>
                  ))}
                  {performance.volatility && typeof performance.volatility === 'object' && Object.entries(performance.volatility).map(([key, value]) => (
                    <div key={`vol-${key}`}>
                      <p className="text-xs text-muted-foreground uppercase">Vol {key}</p>
                      <p className="font-mono text-lg font-bold text-muted-foreground">
                        {((value as number) * 100).toFixed(2)}%
                      </p>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}
        </>
      ) : (
        !loading && <EmptyState message="No regime data is available for this ticker." />
      )}
    </div>
  )
}

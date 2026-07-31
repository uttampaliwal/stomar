import { useState, useMemo } from 'react'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

type SortKey = 'ticker' | 'signal' | 'confidence' | 'price' | 'chg_5d' | 'rsi'
type SortDir = 'asc' | 'desc'

export default function Scanner() {
  const { data, loading, error, refetch } = useApi<any>('/api/scanner/')
  const { data: pipeStatus } = useApi<any>('/api/pipeline/status')
  const [sortKey, setSortKey] = useState<SortKey>('confidence')
  const [sortDir, setSortDir] = useState<SortDir>('desc')

  const sortedResults = useMemo(() => {
    if (!data?.results?.length) return []
    const results = [...data.results]
    results.sort((a: any, b: any) => {
      let av = a[sortKey], bv = b[sortKey]
      if (sortKey === 'signal') { av = a.confidence; bv = b.confidence }
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return results
  }, [data?.results, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <span className="text-muted-foreground ml-1">↕</span>
    return <span className="text-cyan ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  const noModels = pipeStatus && pipeStatus.trained === 0

  return (
    <div className="space-y-6">
      <PageHeader
        title="Stock Scanner"
        description="Multi-stock ML signal scan"
        badge="Live"
      >
        <button
          onClick={refetch}
          disabled={loading}
          className="rounded-lg bg-cyan px-4 py-2 text-sm font-mono font-semibold text-black transition-colors hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Scanning...' : 'Scan Now'}
        </button>
      </PageHeader>

      {/* Info Banner */}
      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          <span className="font-semibold text-foreground">Signal Type:</span> Raw ML prediction (absolute direction — "price goes up or down").
          Not relative to peers. For relative ranking, see the <span className="text-cyan font-semibold">Ranking</span> tab.
          For unified signal, see <span className="text-cyan font-semibold">Consensus</span>.
        </p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {noModels && !loading && (
        <Card className="text-center py-12">
          <p className="text-4xl mb-3">🔍</p>
          <h3 className="text-lg font-bold">No Trained Models</h3>
          <p className="text-sm text-muted-foreground mt-1">Train models in the Predictions tab first, then refresh the scan.</p>
        </Card>
      )}

      {data && !data.error && (
        <>
          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">Scanning</p>
              <p className="font-mono text-2xl font-bold mt-1">
                <span className="text-cyan">{data.scanned || data.results?.length || 0}</span>
                <span className="text-muted-foreground text-sm"> / {data.total || 20}</span>
              </p>
              <p className="text-xs text-muted-foreground mt-0.5">NSE stocks</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">BUY Signals</p>
              <p className="font-mono text-2xl font-bold text-emerald mt-1">{data.buy_count || 0}</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">SELL Signals</p>
              <p className="font-mono text-2xl font-bold text-rose mt-1">{data.sell_count || 0}</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">Models Ready</p>
              <p className="font-mono text-2xl font-bold mt-1">
                <span className="text-cyan">{data.trained || pipeStatus?.trained || 0}</span>
                <span className="text-muted-foreground text-sm"> / {data.total || 20}</span>
              </p>
            </Card>
          </div>

          {/* Results Table */}
          <SectionHeader title="Scan Results" />
          <Card className="overflow-x-auto">
            {sortedResults.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                    <th className="text-left py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('ticker')}>
                      Ticker <SortIcon col="ticker" />
                    </th>
                    <th className="text-center py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('signal')}>
                      Signal <SortIcon col="signal" />
                    </th>
                    <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('confidence')}>
                      Confidence <SortIcon col="confidence" />
                    </th>
                    <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('price')}>
                      Price <SortIcon col="price" />
                    </th>
                    <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('chg_5d')}>
                      5D Chg <SortIcon col="chg_5d" />
                    </th>
                    <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('rsi')}>
                      RSI <SortIcon col="rsi" />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((r: any) => (
                    <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.signal === 'BUY' ? 'success' : 'danger'}>
                          {r.signal === 'BUY' ? '▲' : '▼'} {r.signal}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-right font-mono">{(r.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2.5 text-right font-mono">₹{r.price}</td>
                      <td className={`py-2.5 text-right font-mono ${(r.chg_5d || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                        {(r.chg_5d || 0) >= 0 ? '+' : ''}{r.chg_5d || 0}%
                      </td>
                      <td className={`py-2.5 text-right font-mono ${r.rsi > 70 ? 'text-rose' : r.rsi < 30 ? 'text-emerald' : 'text-muted-foreground'}`}>
                        {r.rsi}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              !loading && !noModels && <EmptyState message="No scan results available" />
            )}
          </Card>
        </>
      )}
    </div>
  )
}

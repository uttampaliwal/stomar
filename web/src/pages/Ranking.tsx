import { useState, useMemo } from 'react'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

type SortKey = 'rank' | 'composite_score' | 'momentum_score' | 'volatility_score' | 'technical_score' | 'ml_score'
type SortDir = 'asc' | 'desc'

export default function Ranking() {
  const { data, loading, error, refetch } = useApi<any>('/api/ranking/')
  const [sortKey, setSortKey] = useState<SortKey>('rank')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const sortedRankings = useMemo(() => {
    if (!data?.rankings?.length) return []
    const rankings = [...data.rankings]
    rankings.sort((a: any, b: any) => {
      let av: any, bv: any
      if (sortKey === 'rank') { av = a.rank; bv = b.rank }
      else if (sortKey === 'composite_score') { av = a.composite_score; bv = b.composite_score }
      else if (sortKey === 'momentum_score') { av = a.momentum?.momentum_combined || 0; bv = b.momentum?.momentum_combined || 0 }
      else if (sortKey === 'volatility_score') { av = a.volatility?.vol_score || 0; bv = b.volatility?.vol_score || 0 }
      else if (sortKey === 'technical_score') { av = a.technical?.technical_combined || 0; bv = b.technical?.technical_combined || 0 }
      else if (sortKey === 'ml_score') { av = a.ml_score || 0; bv = b.ml_score || 0 }
      else { av = a[sortKey] || 0; bv = b[sortKey] || 0 }
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv) : bv.localeCompare(av)
      return sortDir === 'asc' ? av - bv : bv - av
    })
    return rankings
  }, [data?.rankings, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir(key === 'rank' ? 'asc' : 'desc') }
  }

  const SortIcon = ({ col }: { col: SortKey }) => {
    if (sortKey !== col) return <span className="text-muted-foreground ml-1">↕</span>
    return <span className="text-cyan ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Stock Ranking"
        description="Cross-sectional momentum, volatility, technical, and ML composite"
        badge="Relative"
      >
        <button
          onClick={refetch}
          disabled={loading}
          className="rounded-lg bg-cyan px-4 py-2 text-sm font-mono font-semibold text-black transition-colors hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Ranking...' : 'Rank All Stocks'}
        </button>
      </PageHeader>

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          <span className="font-semibold text-foreground">Relative ranking vs peers</span> (percentile).
          For absolute direction, see <span className="text-cyan font-semibold">Scanner</span>.
          For unified signal, see <span className="text-cyan font-semibold">Consensus</span>.
        </p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && data.rankings && (
        <Card className="overflow-x-auto">
          {sortedRankings.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                  <th className="text-left py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('rank')}># <SortIcon col="rank" /></th>
                  <th className="text-left py-2">Ticker</th>
                  <th className="text-right py-2">Price</th>
                  <th className="text-right py-2">Return</th>
                  <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('momentum_score')}>Momentum <SortIcon col="momentum_score" /></th>
                  <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('volatility_score')}>Volatility <SortIcon col="volatility_score" /></th>
                  <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('technical_score')}>Technical <SortIcon col="technical_score" /></th>
                  <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('ml_score')}>ML Score <SortIcon col="ml_score" /></th>
                  <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('composite_score')}>Composite <SortIcon col="composite_score" /></th>
                  <th className="text-center py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {sortedRankings.map((r: any) => (
                  <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                    <td className="py-2.5 font-mono text-muted-foreground">#{r.rank}</td>
                    <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                    <td className="py-2.5 text-right font-mono">₹{r.current_price?.toFixed(2)}</td>
                    <td className={`py-2.5 text-right font-mono ${(r.daily_return || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {((r.daily_return || 0) * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 text-right font-mono">{(r.momentum?.momentum_combined || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono">{(r.volatility?.vol_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono">{(r.technical?.technical_combined || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono">{(r.ml_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono font-bold">{(r.composite_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-center">
                      <Badge variant={
                        r.rank <= 3 ? 'success' :
                        r.rank <= 10 ? 'default' : 'danger'
                      }>
                        {r.rank <= 3 ? 'STRONG BUY' : r.rank <= 10 ? 'HOLD' : 'SELL'}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            !loading && <EmptyState message="No rankings available" />
          )}
        </Card>
      )}
    </div>
  )
}

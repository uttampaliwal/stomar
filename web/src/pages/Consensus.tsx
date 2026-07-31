import { useState, useMemo } from 'react'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import type { ConsensusResponse } from '../lib/api-types'

type SortKey = 'ticker' | 'ensemble_confidence' | 'meta_confidence' | 'consensus'
type SortDir = 'asc' | 'desc'

const SortIcon = ({ col, sortKey, sortDir }: { col: SortKey; sortKey: SortKey; sortDir: SortDir }) => {
  if (sortKey !== col) return <span className="text-muted-foreground ml-1">↕</span>
  return <span className="text-cyan ml-1">{sortDir === 'asc' ? '↑' : '↓'}</span>
}

type BadgeVariant = 'default' | 'success' | 'danger' | 'warning' | 'info' | 'outline'

const CONSENSUS_VARIANT: Record<string, BadgeVariant> = {
  'STRONG BUY': 'success',
  'BUY': 'success',
  'STRONG SELL': 'danger',
  'SELL': 'danger',
  'CONFLICTED': 'warning',
  'HOLD': 'default',
}

const SIGNAL_EMOJI: Record<string, string> = {
  BUY: '🟢',
  SELL: '🔴',
  Bull: '🟢',
  Bear: '🔴',
  Sideways: '🟡',
  'N/A': '⚪',
}

export default function Consensus() {
  const { data, loading, error, refetch } = useApi<ConsensusResponse>('/api/consensus/')
  const [sortKey, setSortKey] = useState<SortKey>('consensus')
  const [sortDir, setSortDir] = useState<SortDir>('asc')

  const sortedResults = useMemo(() => {
    const results = [...(data?.results ?? [])]
    results.sort((a, b) => {
      let av: string | number, bv: string | number
      if (sortKey === 'ticker') { av = a.ticker; bv = b.ticker }
      else if (sortKey === 'consensus') {
        const order: Record<string, number> = { 'STRONG BUY': 0, 'BUY': 1, 'HOLD': 2, 'CONFLICTED': 3, 'SELL': 4, 'STRONG SELL': 5 }
        av = order[a.consensus] ?? 6; bv = order[b.consensus] ?? 6
      }
      else { av = a[sortKey] || 0; bv = b[sortKey] || 0 }
      if (typeof av === 'string') return sortDir === 'asc' ? av.localeCompare(bv as string) : (bv as string).localeCompare(av)
      return sortDir === 'asc' ? av - (bv as number) : (bv as number) - av
    })
    return results
  }, [data?.results, sortKey, sortDir])

  const handleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir(d => d === 'asc' ? 'desc' : 'asc')
    else { setSortKey(key); setSortDir('desc') }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Signal Consensus"
        description="Unified view across all modules — the only signal that matters"
        badge="Consensus"
      >
        <button
          onClick={refetch}
          disabled={loading}
          className="rounded-lg bg-cyan px-4 py-2 text-sm font-mono font-semibold text-black transition-colors hover:opacity-90 disabled:opacity-50"
        >
          {loading ? 'Scanning...' : 'Scan Now'}
        </button>
      </PageHeader>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !('error' in data) && (
        <>
          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">BUY</p>
              <p className="font-mono text-2xl font-bold text-emerald mt-1">{data.buy_count || 0}</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">SELL</p>
              <p className="font-mono text-2xl font-bold text-rose mt-1">{data.sell_count || 0}</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">HOLD</p>
              <p className="font-mono text-2xl font-bold text-muted-foreground mt-1">{data.hold_count || 0}</p>
            </Card>
            <Card className="text-center py-4">
              <p className="text-xs text-muted-foreground uppercase tracking-wider">CONFLICTED</p>
              <p className="font-mono text-2xl font-bold text-amber mt-1">{data.conflicted || 0}</p>
            </Card>
          </div>

          {/* How to Read */}
          <Card className="border-l-4 border-l-cyan bg-cyan/5">
            <p className="text-sm text-muted-foreground">
              <span className="font-semibold text-foreground">How to read:</span> Consensus combines 3 signals:
              Ensemble (ML prediction), Meta-Controller (all 14 modules), and Regime (market direction).
              Meta-Controller is authoritative. If all 3 agree → STRONG BUY/SELL.
              If Meta-Controller agrees with at least 1 other → BUY/SELL.
              If signals conflict → CONFLICTED.
            </p>
          </Card>

          {/* Results Table */}
          <SectionHeader title="Consensus Signals" />
          <Card className="overflow-x-auto">
            {sortedResults.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                    <th className="text-left py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('ticker')}>
                      Stock <SortIcon col="ticker" sortKey={sortKey} sortDir={sortDir} />
                    </th>
                    <th className="text-center py-2">Ensemble</th>
                    <th className="text-center py-2">Meta-Ctrl</th>
                    <th className="text-center py-2">Regime</th>
                    <th className="text-right py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('meta_confidence')}>
                      Confidence <SortIcon col="meta_confidence" sortKey={sortKey} sortDir={sortDir} />
                    </th>
                    <th className="text-center py-2 cursor-pointer hover:text-foreground" onClick={() => handleSort('consensus')}>
                      Consensus <SortIcon col="consensus" sortKey={sortKey} sortDir={sortDir} />
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {sortedResults.map((r) => (
                    <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.ensemble_signal === 'BUY' ? 'success' : r.ensemble_signal === 'SELL' ? 'danger' : 'default'}>
                          {SIGNAL_EMOJI[r.ensemble_signal] || '⚪'} {r.ensemble_signal}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.meta_signal === 'BUY' ? 'success' : r.meta_signal === 'SELL' ? 'danger' : 'default'}>
                          {SIGNAL_EMOJI[r.meta_signal] || '⚪'} {r.meta_signal}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.regime === 'Bull' ? 'success' : r.regime === 'Bear' ? 'danger' : 'warning'}>
                          {SIGNAL_EMOJI[r.regime] || '🟡'} {r.regime}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-right font-mono">
                        {r.meta_confidence > 0 ? `${(r.meta_confidence * 100).toFixed(0)}%` : '—'}
                      </td>
                      <td className="py-2.5 text-center">
                        <Badge variant={CONSENSUS_VARIANT[r.consensus] || 'default'}>
                          {r.consensus}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              !loading && <EmptyState message="No stock data available. Train models first." />
            )}
          </Card>

          {/* Why Modules Disagree */}
          <Card className="border-l-4 border-l-amber bg-amber/5">
            <p className="text-sm text-muted-foreground">
              <span className="font-semibold text-foreground">Why modules disagree:</span> Scanner shows raw ML prediction (absolute direction).
              Ranking shows relative quality vs peers (a stock can be #1 ranked but still predicted to fall).
              Risk shows market regime (portfolio-level, not per-stock).
              <span className="font-semibold text-foreground"> Only the Meta-Controller combines all signals.</span>
            </p>
          </Card>
        </>
      )}
    </div>
  )
}

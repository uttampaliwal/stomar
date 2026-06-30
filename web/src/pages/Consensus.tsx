import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Consensus() {
  const { data, loading, error } = useApi<any>('/api/consensus/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Consensus</h1>
        <p className="text-sm text-muted-foreground">14-signal unified consensus view</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Strong Buy" value={data.buy_count || 0} trend="up" />
            <Stat label="Sell" value={data.sell_count || 0} trend="down" />
            <Stat label="Hold" value={data.hold_count || 0} />
            <Stat label="Conflicted" value={data.conflicted || 0} trend="neutral" />
          </div>

          <SectionHeader title="Consensus Signals" />
          <Card className="overflow-x-auto">
            {data.results?.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                    <th className="text-left py-2">Stock</th>
                    <th className="text-center py-2">Ensemble</th>
                    <th className="text-center py-2">Regime</th>
                    <th className="text-right py-2">Sentiment</th>
                    <th className="text-center py-2">Consensus</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r: any) => (
                    <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.ensemble_signal === 'BUY' ? 'success' : r.ensemble_signal === 'SELL' ? 'danger' : 'default'}>
                          {r.ensemble_signal}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.regime === 'Bull' ? 'success' : r.regime === 'Bear' ? 'danger' : 'warning'}>
                          {r.regime}
                        </Badge>
                      </td>
                      <td className="py-2.5 text-right font-mono text-muted-foreground">{r.sentiment?.toFixed(2)}</td>
                      <td className="py-2.5 text-center">
                        <Badge variant={
                          r.consensus?.includes('BUY') ? 'success' :
                          r.consensus?.includes('SELL') ? 'danger' :
                          r.consensus === 'HOLD' ? 'warning' : 'default'
                        }>
                          {r.consensus}
                        </Badge>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <EmptyState message="No consensus data available" />
            )}
          </Card>
        </>
      )}
    </div>
  )
}

import { Card, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Ranking() {
  const { data, loading, error } = useApi<any>('/api/ranking/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Stock Ranking</h1>
        <p className="text-sm text-muted-foreground">Cross-sectional momentum, volatility, technical, ML composite</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && data.rankings && (
        <Card className="overflow-x-auto">
          {data.rankings.length > 0 ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                  <th className="text-left py-2">Rank</th>
                  <th className="text-left py-2">Ticker</th>
                  <th className="text-right py-2">Price</th>
                  <th className="text-right py-2">Return</th>
                  <th className="text-right py-2">Momentum</th>
                  <th className="text-right py-2">Volatility</th>
                  <th className="text-right py-2">Technical</th>
                  <th className="text-right py-2">Composite</th>
                  <th className="text-center py-2">Action</th>
                </tr>
              </thead>
              <tbody>
                {data.rankings.map((r: any) => (
                  <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                    <td className="py-2.5 font-mono text-muted-foreground">#{r.rank}</td>
                    <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                    <td className="py-2.5 text-right font-mono">₹{r.price?.toFixed(2)}</td>
                    <td className={`py-2.5 text-right font-mono ${(r.daily_return || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {((r.daily_return || 0) * 100).toFixed(2)}%
                    </td>
                    <td className="py-2.5 text-right font-mono">{(r.momentum_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono">{(r.volatility_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono">{(r.technical_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-right font-mono font-bold">{(r.composite_score || 0).toFixed(3)}</td>
                    <td className="py-2.5 text-center">
                      <Badge variant={
                        r.recommendation?.includes('BUY') ? 'success' :
                        r.recommendation?.includes('SELL') ? 'danger' : 'warning'
                      }>
                        {r.recommendation}
                      </Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <EmptyState message="No rankings available" />
          )}
        </Card>
      )}
    </div>
  )
}

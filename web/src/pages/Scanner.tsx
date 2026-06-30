import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Scanner() {
  const { data, loading, error } = useApi<any>('/api/scanner/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Stock Scanner</h1>
        <p className="text-sm text-muted-foreground">Multi-stock ML signal scan</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && (
        <>
          <div className="grid grid-cols-2 gap-3">
            <Stat label="BUY Signals" value={data.buy_count || 0} trend="up" />
            <Stat label="SELL Signals" value={data.sell_count || 0} trend="down" />
          </div>

          <SectionHeader title="Scan Results" />
          <Card className="overflow-x-auto">
            {data.results?.length > 0 ? (
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                    <th className="text-left py-2">Ticker</th>
                    <th className="text-center py-2">Signal</th>
                    <th className="text-right py-2">Confidence</th>
                    <th className="text-right py-2">Price</th>
                    <th className="text-right py-2">Day Return</th>
                    <th className="text-right py-2">RSI</th>
                  </tr>
                </thead>
                <tbody>
                  {data.results.map((r: any) => (
                    <tr key={r.ticker} className="border-b border-border/50 hover:bg-accent/30">
                      <td className="py-2.5 font-mono font-semibold">{r.ticker}</td>
                      <td className="py-2.5 text-center">
                        <Badge variant={r.signal === 'BUY' ? 'success' : 'danger'}>{r.signal}</Badge>
                      </td>
                      <td className="py-2.5 text-right font-mono">{(r.confidence * 100).toFixed(1)}%</td>
                      <td className="py-2.5 text-right font-mono">₹{r.price}</td>
                      <td className={`py-2.5 text-right font-mono ${r.day_return >= 0 ? 'text-emerald' : 'text-rose'}`}>
                        {r.day_return >= 0 ? '+' : ''}{r.day_return}%
                      </td>
                      <td className="py-2.5 text-right font-mono text-muted-foreground">{r.rsi}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : (
              <EmptyState message="No scan results available" />
            )}
          </Card>
        </>
      )}
    </div>
  )
}

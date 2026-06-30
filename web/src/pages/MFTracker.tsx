import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'

export default function MFTracker() {
  const { data, loading, error } = useApi<any>('/api/mf-tracker/portfolio')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">MF Tracker</h1>
        <p className="text-sm text-muted-foreground">Mutual fund portfolio with XIRR and factor analysis</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Invested" value={formatCurrency(data.invested || 0)} />
            <Stat label="Current Value" value={formatCurrency(data.current_value || 0)} />
            <Stat label="P&L" value={formatCurrency(data.pnl || 0)} trend={data.pnl >= 0 ? 'up' : 'down'} />
            <Stat label="Return" value={`${(data.return_pct || 0).toFixed(2)}%`} trend={data.return_pct >= 0 ? 'up' : 'down'} />
          </div>

          {/* Holdings */}
          {data.holdings?.length > 0 && (
            <>
              <SectionHeader title="Holdings" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                      <th className="text-left py-2">Fund</th>
                      <th className="text-right py-2">Units</th>
                      <th className="text-right py-2">Avg NAV</th>
                      <th className="text-right py-2">Current</th>
                      <th className="text-right py-2">P&L</th>
                      <th className="text-right py-2">Return</th>
                    </tr>
                  </thead>
                  <tbody>
                    {data.holdings.map((h: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                        <td className="py-2.5 font-semibold">{h.fund_name || h.ticker}</td>
                        <td className="py-2.5 text-right font-mono">{h.units?.toFixed(2)}</td>
                        <td className="py-2.5 text-right font-mono">₹{h.avg_nav?.toFixed(2)}</td>
                        <td className="py-2.5 text-right font-mono">{formatCurrency(h.current_value || 0)}</td>
                        <td className={`py-2.5 text-right font-mono ${(h.pnl || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {formatCurrency(h.pnl || 0)}
                        </td>
                        <td className={`py-2.5 text-right font-mono ${(h.return_pct || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {(h.return_pct || 0).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}

          {/* Concentration Risk */}
          {data.concentration_risk?.length > 0 && (
            <>
              <SectionHeader title="Concentration Risk" />
              {data.concentration_risk.map((r: any, i: number) => (
                <Card key={i} className="border-amber/30 bg-amber/5">
                  <p className="text-sm text-amber">{r.message || `${r.fund_name || r.ticker}: ${(r.weight * 100).toFixed(1)}%`}</p>
                </Card>
              ))}
            </>
          )}
        </>
      )}
    </div>
  )
}

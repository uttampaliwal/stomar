import { Card, Spinner, ErrorDisplay, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

function getCorrelationColor(val: number): string {
  if (val === 1) return 'bg-cyan/20'
  if (val > 0.7) return 'bg-emerald/40'
  if (val > 0.5) return 'bg-emerald/30'
  if (val > 0.3) return 'bg-emerald/20'
  if (val > 0) return 'bg-emerald/10'
  if (val > -0.3) return 'bg-rose/10'
  if (val > -0.5) return 'bg-rose/20'
  if (val > -0.7) return 'bg-rose/30'
  return 'bg-rose/40'
}

export default function Correlation() {
  const { data, loading, error } = useApi<any>('/api/correlation/')

  return (
    <div className="space-y-6">
      <PageHeader
        title="Correlation Matrix"
        description="Cross-asset correlation heatmap"
        badge="Diversification"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">Use this view to understand how assets move together and whether the portfolio is over-concentrated.</p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && data.tickers ? (
        <>
          <Card className="overflow-x-auto">
            <div className="min-w-[600px]">
              <table className="w-full text-xs font-mono">
                <thead>
                  <tr>
                    <th className="p-2"></th>
                    {data.tickers.map((t: string) => (
                      <th key={t} className="p-2 text-center text-muted-foreground">{t}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {data.tickers.map((rowTicker: string, i: number) => (
                    <tr key={rowTicker}>
                      <td className="p-2 text-muted-foreground font-semibold">{rowTicker}</td>
                      {data.matrix[i].map((val: number, j: number) => (
                        <td key={j} className={`p-2 text-center font-mono font-bold ${getCorrelationColor(val)} ${i === j ? 'ring-1 ring-cyan/30' : ''}`}>
                          {val.toFixed(2)}
                        </td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Legend */}
          <Card>
            <div className="flex items-center justify-center gap-4 text-xs">
              <span className="text-rose font-bold">-1.0</span>
              <div className="flex gap-0.5">
                {['bg-rose/40', 'bg-rose/30', 'bg-rose/20', 'bg-rose/10', 'bg-transparent', 'bg-emerald/10', 'bg-emerald/20', 'bg-emerald/30', 'bg-emerald/40'].map((c, i) => (
                  <div key={i} className={`w-6 h-3 rounded ${c}`} />
                ))}
              </div>
              <span className="text-emerald font-bold">+1.0</span>
            </div>
          </Card>

          {data.diversification_score !== undefined && (
            <Card>
              <div className="text-center">
                <p className="text-xs text-muted-foreground uppercase">Diversification Score</p>
                <p className="font-mono text-3xl font-bold text-cyan">{data.diversification_score.toFixed(1)}%</p>
              </div>
            </Card>
          )}
        </>
      ) : (
        !loading && <EmptyState message="No correlation matrix data is available yet." />
      )}
    </div>
  )
}

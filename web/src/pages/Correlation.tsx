import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Correlation() {
  const { data, loading, error } = useApi<any>('/api/correlation/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Correlation Matrix</h1>
        <p className="text-sm text-muted-foreground">Cross-asset correlation heatmap</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && data.tickers && (
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
                    {data.matrix[i].map((val: number, j: number) => {
                      const abs = Math.abs(val)
                      const bg = i === j ? 'bg-cyan/20' :
                        val > 0.7 ? 'bg-emerald/30' :
                        val > 0.3 ? 'bg-emerald/15' :
                        val < -0.3 ? 'bg-rose/15' :
                        val < -0.7 ? 'bg-rose/30' : ''
                      return (
                        <td key={j} className={`p-2 text-center ${bg}`}>
                          {val.toFixed(2)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Card>
      )}
    </div>
  )
}

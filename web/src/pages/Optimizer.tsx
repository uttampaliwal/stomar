import { Card, Stat, SectionHeader, Spinner, ErrorDisplay } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Optimizer() {
  const { data, loading, error } = useApi<any>('/api/optimizer/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Portfolio Optimizer</h1>
        <p className="text-sm text-muted-foreground">Max Sharpe, Min Variance, Black-Litterman</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          {(['max_sharpe', 'min_variance', 'black_litterman'] as const).map((key) => {
            const p = data[key]
            if (!p) return null
            const name = key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
            const weights: Record<string, number> = {}
            if (p.weights && p.tickers) {
              const wArr = Array.isArray(p.weights) ? p.weights : Object.values(p.weights)
              p.tickers.forEach((t: string, i: number) => { weights[t] = Number(wArr[i]) })
            } else if (p.weights && typeof p.weights === 'object') {
              Object.assign(weights, p.weights)
            }
            const expReturn = p.expected_return ?? p.return ?? 0
            const vol = p.volatility ?? 0
            return (
              <Card key={key}>
                <SectionHeader title={name} />
                <div className="grid grid-cols-3 gap-3">
                  <Stat label="Expected Return" value={`${(expReturn * 100).toFixed(2)}%`} />
                  <Stat label="Volatility" value={`${(vol * 100).toFixed(2)}%`} />
                  <Stat label="Sharpe" value={(p.sharpe || 0).toFixed(2)} />
                </div>
                {Object.keys(weights).length > 0 && (
                  <div className="mt-3 flex flex-wrap gap-1.5">
                    {Object.entries(weights).map(([ticker, weight]) => (
                      <span key={ticker} className="rounded-md bg-cyan/10 px-2 py-0.5 text-xs font-mono text-cyan border border-cyan/20">
                        {ticker}: {(weight * 100).toFixed(1)}%
                      </span>
                    ))}
                  </div>
                )}
              </Card>
            )
          })}
        </>
      )}
    </div>
  )
}

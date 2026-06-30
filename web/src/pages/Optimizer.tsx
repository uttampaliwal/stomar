import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, EmptyState } from '@/components/UI'
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
          {data.optimal_portfolios && Object.entries(data.optimal_portfolios).map(([name, portfolio]: [string, any]) => (
            <Card key={name}>
              <SectionHeader title={name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())} />
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Expected Return" value={`${((portfolio.expected_return || 0) * 100).toFixed(2)}%`} />
                <Stat label="Volatility" value={`${((portfolio.volatility || 0) * 100).toFixed(2)}%`} />
                <Stat label="Sharpe" value={(portfolio.sharpe || 0).toFixed(2)} />
              </div>
              {portfolio.weights && (
                <div className="mt-3 flex flex-wrap gap-1.5">
                  {Object.entries(portfolio.weights).map(([ticker, weight]) => (
                    <span key={ticker} className="rounded-md bg-cyan/10 px-2 py-0.5 text-xs font-mono text-cyan border border-cyan/20">
                      {ticker}: {((Number(weight)) * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              )}
            </Card>
          ))}
        </>
      )}
    </div>
  )
}

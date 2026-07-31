import { useMemo } from 'react'
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceDot } from 'recharts'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import type { OptimizerResponse } from '../lib/api-types'

type StrategyData = {
  weights?: number[] | Record<string, number>
  tickers?: string[]
  return?: number
  expected_return?: number
  volatility?: number
  sharpe?: number
}
type FrontierPoint = OptimizerResponse['efficient_frontier'][number] & { sharpe?: number }

export default function Optimizer() {
  const { data, loading, error } = useApi<OptimizerResponse>('/api/optimizer/')

  const frontierData = useMemo(() => {
    return data?.efficient_frontier?.map((p: FrontierPoint) => ({
      return: (p.return || 0) * 100,
      vol: (p.volatility || 0) * 100,
      sharpe: p.sharpe || 0,
    })) || []
  }, [data?.efficient_frontier])

  const strategies = [
    { key: 'max_sharpe', label: 'Max Sharpe', color: 'text-emerald' },
    { key: 'min_variance', label: 'Min Variance', color: 'text-cyan' },
    { key: 'black_litterman', label: 'Black-Litterman', color: 'text-amber' },
  ] as const

  return (
    <div className="space-y-6">
      <PageHeader
        title="Portfolio Optimizer"
        description="Max Sharpe, Min Variance, and Black-Litterman"
        badge="Allocation"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">These portfolios are optimized from the same market context that drives the risk and automation views.</p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !('error' in data) && (
        <>
          {/* Efficient Frontier */}
          {frontierData.length > 0 && (
            <>
              <SectionHeader title="Efficient Frontier" />
              <Card className="overflow-hidden">
                <ResponsiveContainer width="100%" height={300}>
                  <ScatterChart margin={{ top: 5, right: 5, bottom: 5, left: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis type="number" dataKey="vol" name="Volatility %" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} label={{ value: 'Volatility %', position: 'bottom', fontSize: 10 }} />
                    <YAxis type="number" dataKey="return" name="Return %" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))' }} label={{ value: 'Return %', angle: -90, position: 'insideLeft', fontSize: 10 }} />
                    <Tooltip contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: '8px', fontSize: '11px' }} formatter={(v: number, name: string) => [`${v.toFixed(2)}%`, name]} />
                    <Scatter data={frontierData} fill="#06b6d4" fillOpacity={0.6} />
                    {/* Mark optimal portfolios */}
                    {strategies.map(s => {
                      const p = data[s.key] as unknown as StrategyData
                      if (!p) return null
                      return (
                        <ReferenceDot
                          key={s.key}
                          x={(p.volatility || 0) * 100}
                          y={(p.return || p.expected_return || 0) * 100}
                          r={6}
                          fill={s.key === 'max_sharpe' ? '#10b981' : s.key === 'min_variance' ? '#06b6d4' : '#f59e0b'}
                          stroke="white"
                          strokeWidth={2}
                        />
                      )
                    })}
                  </ScatterChart>
                </ResponsiveContainer>
                <div className="flex justify-center gap-4 mt-2 text-xs">
                  {strategies.map(s => (
                    <span key={s.key} className="flex items-center gap-1">
                      <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: s.key === 'max_sharpe' ? '#10b981' : s.key === 'min_variance' ? '#06b6d4' : '#f59e0b' }} />
                      {s.label}
                    </span>
                  ))}
                </div>
              </Card>
            </>
          )}

          {strategies.map(({ key, label }) => {
            const p = data[key] as unknown as StrategyData
            if (!p) return null
            const weights: Record<string, number> = {}
            if (p.weights && p.tickers) {
              const wArr = Array.isArray(p.weights) ? p.weights : Object.values(p.weights)
              p.tickers.forEach((t: string, i: number) => { weights[t] = Number(wArr[i]) })
            }
            return (
              <Card key={key}>
                <SectionHeader title={label} />
                <div className="grid grid-cols-3 gap-3">
                  <Stat label="Expected Return" value={`${((p.expected_return ?? p.return ?? 0) * 100).toFixed(2)}%`} />
                  <Stat label="Volatility" value={`${((p.volatility ?? 0) * 100).toFixed(2)}%`} />
                  <Stat label="Sharpe" value={(p.sharpe || 0).toFixed(2)} />
                </div>
                {Object.keys(weights).length > 0 && (
                  <div className="mt-3 space-y-1.5">
                    {Object.entries(weights)
                      .sort(([, a], [, b]) => b - a)
                      .map(([ticker, weight]) => (
                      <div key={ticker} className="flex items-center gap-2">
                        <span className="text-xs font-mono w-32 truncate">{ticker}</span>
                        <div className="flex-1 bg-secondary rounded-full h-2">
                          <div
                            className="h-2 rounded-full bg-cyan"
                            style={{ width: `${Math.min(weight * 100 / 0.4 * 100, 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono text-muted-foreground w-14 text-right">{(weight * 100).toFixed(1)}%</span>
                      </div>
                    ))}
                  </div>
                )}
              </Card>
            )
          })}

          {(!data?.efficient_frontier?.length || !strategies.some(s => !!data[s.key])) && (
            <EmptyState message="No optimizer output is available yet." />
          )}
        </>
      )}
    </div>
  )
}

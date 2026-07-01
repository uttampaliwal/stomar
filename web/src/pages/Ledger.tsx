import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Ledger() {
  const { data, loading, error, refetch } = useApi<any>('/api/ledger/summary')
  const { data: decisionsData } = useApi<any>('/api/ledger/decisions')
  const { data: performance } = useApi<any>('/api/ledger/performance')
  const [running, setRunning] = useState(false)
  const [runResult, setRunResult] = useState<string | null>(null)

  const handleRunPipeline = async () => {
    setRunning(true)
    setRunResult(null)
    try {
      const res = await fetch('/api/pipeline/run', { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') {
        setRunResult(`Pipeline started for ${data.tickers?.length || 0} tickers. Monitoring...`)
        // Poll for completion
        const poll = setInterval(async () => {
          const statusRes = await fetch('/api/pipeline/run/status')
          const status = await statusRes.json()
          if (status.status === 'done') {
            clearInterval(poll)
            const d = status.result
            setRunResult(
              `Complete: ${d?.decisions?.length || 0} decisions, ` +
              `${d?.trades?.length || 0} trades, ${d?.errors?.length || 0} errors`
            )
            setRunning(false)
            refetch()
          } else if (status.status === 'error') {
            clearInterval(poll)
            setRunResult(`Error: ${status.error}`)
            setRunning(false)
          }
        }, 3000)
      } else if (data.status === 'already_running') {
        setRunResult('Pipeline already running...')
      } else if (data.status === 'no_trained_models') {
        setRunResult('No trained models found. Train models first from Pipeline page.')
        setRunning(false)
      } else {
        setRunResult(JSON.stringify(data))
        setRunning(false)
      }
    } catch (e: any) {
      setRunResult(`Failed: ${e.message}`)
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Trading Ledger</h1>
          <p className="text-sm text-muted-foreground">Decision history, signal accuracy, P&L tracking</p>
        </div>
        <button
          onClick={handleRunPipeline}
          disabled={running}
          className="px-4 py-2 rounded-lg text-sm font-medium bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50"
        >
          {running ? 'Running...' : 'Run Pipeline'}
        </button>
      </div>

      {runResult && (
        <div className="p-3 rounded-lg bg-muted text-sm font-mono">{runResult}</div>
      )}

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Total Decisions" value={data.total_decisions || 0} />
            <Stat label="Total Trades" value={data.total_trades || 0} />
            <Stat label="Accuracy" value={`${((data.accuracy || 0) * 100).toFixed(1)}%`} trend={data.accuracy > 0.5 ? 'up' : 'down'} />
            <Stat label="Signal Types" value={Object.keys(data.signal_accuracy || {}).length} />
          </div>

          {/* Signal Accuracy */}
          {data.signal_accuracy && Object.keys(data.signal_accuracy).length > 0 && (
            <>
              <SectionHeader title="Signal Module Accuracy" />
              <Card>
                <div className="space-y-2">
                  {Object.entries(data.signal_accuracy).map(([module, acc]: [string, any]) => (
                    <div key={module} className="flex items-center gap-3">
                      <span className="w-32 text-xs font-mono uppercase text-muted-foreground truncate">{module}</span>
                      <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all ${(acc || 0) > 0.6 ? 'bg-emerald' : (acc || 0) > 0.4 ? 'bg-amber' : 'bg-rose'}`}
                          style={{ width: `${(acc || 0) * 100}%` }}
                        />
                      </div>
                      <span className="font-mono text-xs w-12 text-right">{((acc || 0) * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </Card>
            </>
          )}

          {/* P&L Curve */}
          {data.snapshots?.length > 0 && (
            <>
              <SectionHeader title="Portfolio Value Over Time" />
              <Card className="overflow-x-auto">
                <div className="flex gap-1 items-end h-32">
                  {data.snapshots.map((s: any, i: number) => {
                    const val = s.total_value || 0
                    const maxVal = Math.max(...data.snapshots.map((x: any) => x.total_value || 0))
                    const height = maxVal > 0 ? (val / maxVal) * 100 : 0
                    return (
                      <div
                        key={i}
                        className="flex-1 bg-gradient-to-t from-cyan/60 to-cyan/20 rounded-t-sm min-w-[4px]"
                        style={{ height: `${height}%` }}
                        title={`${s.date}: ₹${val.toLocaleString()}`}
                      />
                    )
                  })}
                </div>
                <div className="flex justify-between text-xs text-muted-foreground mt-2">
                  <span>{data.snapshots[0]?.date}</span>
                  <span>{data.snapshots[data.snapshots.length - 1]?.date}</span>
                </div>
              </Card>
            </>
          )}

          {/* Decisions Table */}
          {decisionsData?.decisions?.length > 0 && (
            <>
              <SectionHeader title="Recent Decisions" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Date</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Ticker</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Action</th>
                      <th className="text-right py-2 px-3 font-mono text-xs text-muted-foreground">Confidence</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Regime</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Reasoning</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisionsData.decisions.slice(0, 20).map((d: any, i: number) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                        <td className="py-2 px-3 font-mono text-xs">{d.date}</td>
                        <td className="py-2 px-3 font-mono text-xs">{d.ticker}</td>
                        <td className="py-2 px-3">
                          <Badge variant={d.action === 'BUY' ? 'success' : d.action === 'SELL' ? 'danger' : 'default'}>
                            {d.action}
                          </Badge>
                        </td>
                        <td className="py-2 px-3 font-mono text-xs text-right">
                          {d.confidence ? `${(d.confidence * 100).toFixed(1)}%` : '-'}
                        </td>
                        <td className="py-2 px-3 text-xs">{d.regime || '-'}</td>
                        <td className="py-2 px-3 text-xs text-muted-foreground max-w-[200px] truncate">{d.reasoning}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}

import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Ledger() {
  const { data, loading, error } = useApi<any>('/api/ledger/summary')
  const { data: performance } = useApi<any>('/api/ledger/performance')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Trading Ledger</h1>
        <p className="text-sm text-muted-foreground">Decision history, signal accuracy, P&L tracking</p>
      </div>

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
        </>
      )}
    </div>
  )
}

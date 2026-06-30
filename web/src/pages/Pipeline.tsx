import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Pipeline() {
  const { data, loading, error } = useApi<any>('/api/pipeline/status')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Training Pipeline</h1>
        <p className="text-sm text-muted-foreground">Model training status and pipeline overview</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Total Stocks" value={data.total || 0} />
            <Stat label="Trained" value={data.trained || 0} trend="up" />
            <Stat label="Pending" value={data.missing || 0} trend={data.missing > 0 ? 'down' : 'up'} />
          </div>

          {/* Progress Bar */}
          <Card>
            <div className="w-full h-3 rounded-full bg-muted overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-cyan to-emerald transition-all"
                style={{ width: `${((data.trained || 0) / (data.total || 1)) * 100}%` }}
              />
            </div>
            <p className="text-xs text-muted-foreground mt-2 text-center">
              {((data.trained || 0) / (data.total || 1) * 100).toFixed(0)}% complete
            </p>
          </Card>

          {/* Model Grid */}
          <SectionHeader title="Model Status" />
          <div className="grid grid-cols-4 md:grid-cols-5 lg:grid-cols-6 gap-2">
            {data.trained_tickers?.map((t: string) => (
              <div key={t} className="rounded-lg bg-emerald/10 border border-emerald/20 p-2 text-center">
                <p className="font-mono text-xs text-emerald">{t}</p>
              </div>
            ))}
            {data.missing_tickers?.map((t: string) => (
              <div key={t} className="rounded-lg bg-muted border border-border p-2 text-center">
                <p className="font-mono text-xs text-muted-foreground">{t}</p>
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  )
}

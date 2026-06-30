import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Monitoring() {
  const { data, loading, error } = useApi<any>('/api/monitoring/')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">System Monitoring</h1>
        <p className="text-sm text-muted-foreground">Model health, data validation, alerts</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && (
        <>
          <div className="grid grid-cols-3 gap-3">
            <Stat label="Models Trained" value={data.models_trained?.length || 0} trend="up" />
            <Stat label="Models Missing" value={data.models_missing?.length || 0} trend={data.models_missing?.length > 0 ? 'down' : 'up'} />
            <Stat label="Total Alerts" value={(data.alerts?.critical?.length || 0) + (data.alerts?.warning?.length || 0)} />
          </div>

          {/* Alerts */}
          {data.alerts?.critical?.length > 0 && (
            <>
              <SectionHeader title="Critical Alerts" />
              {data.alerts.critical.map((a: any, i: number) => (
                <Card key={i} className="border-rose/30 bg-rose/5">
                  <Badge variant="danger" className="mb-2">CRITICAL</Badge>
                  <p className="text-sm">{a.message}</p>
                </Card>
              ))}
            </>
          )}

          {data.alerts?.warning?.length > 0 && (
            <>
              <SectionHeader title="Warnings" />
              {data.alerts.warning.map((a: any, i: number) => (
                <Card key={i} className="border-amber/30 bg-amber/5">
                  <Badge variant="warning" className="mb-2">WARNING</Badge>
                  <p className="text-sm">{a.message}</p>
                </Card>
              ))}
            </>
          )}

          {/* Alert History */}
          {data.alert_history?.length > 0 && (
            <>
              <SectionHeader title="Recent Alert History" />
              <div className="space-y-2">
                {data.alert_history.map((a: any, i: number) => (
                  <Card key={i}>
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <Badge variant={a.severity === 'critical' ? 'danger' : a.severity === 'warning' ? 'warning' : 'info'}>
                          {a.severity}
                        </Badge>
                        <span className="text-sm">{a.message}</span>
                      </div>
                      <span className="text-xs font-mono text-muted-foreground">{a.timestamp}</span>
                    </div>
                  </Card>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

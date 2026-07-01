import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Monitoring() {
  const { data, loading, error } = useApi<any>('/api/monitoring/')

  return (
    <div className="space-y-6">
      <PageHeader
        title="System Monitoring"
        description="Model health, data validation, and alerts"
        badge="Ops"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">This view surfaces the health of the training pipeline and the alert history so issues are visible before they become costly.</p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data ? (
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
      ) : (
        !loading && <EmptyState message="No monitoring data is available right now." />
      )}
    </div>
  )
}

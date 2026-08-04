import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, PageHeader, EmptyState } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import type { MonitoringResponse, DailyHealthResponse, ReadinessResponse } from '../lib/api-types'

type AlertHistoryEntry = { severity?: string; message?: string; timestamp?: string }

export default function Monitoring() {
  const { data, loading, error } = useApi<MonitoringResponse>('/api/monitoring/')
  const { data: health } = useApi<DailyHealthResponse>('/api/monitoring/daily-health')
  const { data: readiness } = useApi<ReadinessResponse>('/api/monitoring/readiness')

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

      {/* P5.2: Daily health sweep */}
      {health && !('error' in health) && (
        <Card className="border-l-4 border-l-emerald bg-emerald/5">
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold">Daily Health Sweep (P5.2)</p>
            <Badge variant={health.status === 'running' ? 'success' : 'warning'}>{health.status}</Badge>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
            <div>
              <p className="text-muted-foreground">Last run</p>
              <p className="font-mono font-semibold">{health.last_run_date ?? 'never'}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Tickers processed</p>
              <p className="font-mono font-semibold">{health.tickers_processed}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Paper trading days</p>
              <p className="font-mono font-semibold">{health.paper_days}</p>
            </div>
            <div>
              <p className="text-muted-foreground">Kill switch</p>
              <p className={`font-mono font-semibold ${health.kill_switch_active ? 'text-rose' : 'text-emerald'}`}>
                {health.kill_switch_active ? 'ACTIVE' : 'OFF'}
              </p>
            </div>
          </div>
          {health.data_freshness_days && Object.keys(health.data_freshness_days).length > 0 && (
            <div className="mt-3">
              <p className="text-xs text-muted-foreground mb-1">Data freshness per ticker (days since last fetch)</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(health.data_freshness_days).map(([ticker, days]) => (
                  <span key={ticker} className={`rounded px-2 py-0.5 font-mono text-xs border ${days != null && days <= 3 ? 'border-emerald/30 bg-emerald/10 text-emerald' : 'border-rose/30 bg-rose/10 text-rose'}`}>
                    {ticker} {days != null ? `${days}d` : '—'}
                  </span>
                ))}
              </div>
            </div>
          )}
          {health.model_age_days && Object.keys(health.model_age_days).length > 0 && (
            <div className="mt-2">
              <p className="text-xs text-muted-foreground mb-1">Model age per ticker (days since last training)</p>
              <div className="flex flex-wrap gap-2">
                {Object.entries(health.model_age_days).map(([ticker, days]) => (
                  <span key={ticker} className={`rounded px-2 py-0.5 font-mono text-xs border ${days != null && days <= 30 ? 'border-emerald/30 bg-emerald/10 text-emerald' : 'border-amber/30 bg-amber/10 text-amber'}`}>
                    {ticker} {days != null ? `${days}d` : '—'}
                  </span>
                ))}
              </div>
            </div>
          )}
        </Card>
      )}

      {/* P5.2: Readiness gate summary */}
      {readiness && !('error' in readiness) && readiness.gates && (
        <Card className={`border-l-4 ${readiness.ready ? 'border-l-emerald bg-emerald/5' : 'border-l-amber bg-amber/5'}`}>
          <div className="flex items-center justify-between mb-2">
            <p className="text-sm font-semibold">Live-Trading Readiness Gate</p>
            <Badge variant={readiness.ready ? 'success' : 'warning'}>
              {readiness.ready ? 'READY' : 'PAPER PHASE'}
            </Badge>
          </div>
          <div className="flex flex-wrap gap-2">
            {Object.entries(readiness.gates).map(([name, gate]) => (
              <span key={name} className={`rounded border px-2 py-0.5 font-mono text-xs ${gate.passed ? 'border-emerald/30 bg-emerald/10 text-emerald' : 'border-rose/30 bg-rose/10 text-rose'}`}>
                {gate.passed ? '✓' : '✗'} {name.replace('_', ' ')}
              </span>
            ))}
          </div>
        </Card>
      )}

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
              {data.alerts.critical.map((a, i: number) => (
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
              {data.alerts.warning.map((a, i: number) => (
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
                {data.alert_history.map((a: AlertHistoryEntry, i: number) => (
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

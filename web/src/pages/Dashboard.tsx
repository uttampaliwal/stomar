import { Card, Stat, SectionHeader, Spinner, ErrorDisplay } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatCurrency, formatPercent } from '@/lib/utils'
import { Brain, TrendingUp, Shield, Zap, Activity, Target } from 'lucide-react'

export default function Dashboard() {
  const marketStatus = useApi<{ status: string }>('/api/market/status')
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const pipeline = useApi<any>('/api/pipeline/status')
  const monitoring = useApi<any>('/api/monitoring/')

  return (
    <div className="space-y-6">
      {/* Hero Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Command Center</h1>
          <p className="text-sm text-muted-foreground mt-1">Quantitative intelligence terminal for NSE markets</p>
        </div>
        <div className="flex items-center gap-2">
          <div className={`h-2 w-2 rounded-full ${marketStatus.data?.status === 'Open' ? 'bg-emerald animate-pulse' : 'bg-rose'}`} />
          <span className="font-mono text-xs text-muted-foreground">{marketStatus.data?.status || 'Loading...'}</span>
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <Stat
          label="Stocks Tracked"
          value={pipeline.data?.total || '—'}
          sub={`${pipeline.data?.trained || 0} trained`}
          trend="up"
        />
        <Stat
          label="Models Ready"
          value={pipeline.data?.trained || '—'}
          sub={`of ${pipeline.data?.total || 20}`}
          trend={pipeline.data?.trained > 0 ? 'up' : 'neutral'}
        />
        <Stat
          label="Critical Alerts"
          value={monitoring.data?.alerts?.critical?.length || 0}
          sub={monitoring.data?.alerts?.warning?.length ? `${monitoring.data.alerts.warning.length} warnings` : 'All clear'}
          trend={monitoring.data?.alerts?.critical?.length > 0 ? 'down' : 'up'}
        />
        <Stat
          label="System Status"
          value="Active"
          sub="FastAPI + React"
          trend="up"
        />
      </div>

      {/* Quick Access Grid */}
      <SectionHeader title="Quick Access" />
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-3">
        {[
          { icon: Brain, label: 'Predictions', desc: 'ML ensemble signals', path: '/predictions', color: 'text-violet' },
          { icon: Target, label: 'Consensus', desc: '14-signal consensus', path: '/consensus', color: 'text-cyan' },
          { icon: Shield, label: 'Risk Analysis', desc: 'VaR, Kelly, drawdown', path: '/risk', color: 'text-rose' },
          { icon: Activity, label: 'Market Pulse', desc: 'FII/DII flows, PCR', path: '/market-pulse', color: 'text-emerald' },
          { icon: TrendingUp, label: 'Optimizer', desc: 'Portfolio allocation', path: '/optimizer', color: 'text-amber' },
          { icon: Zap, label: 'Paper Trading', desc: 'Simulated execution', path: '/paper-trading', color: 'text-cyan' },
        ].map((item) => (
          <a
            key={item.path}
            href={item.path}
            className="group flex items-center gap-3 rounded-xl border border-border bg-card/50 p-4 transition-all hover:border-cyan/30 hover:bg-accent/50"
          >
            <item.icon className={`h-8 w-8 ${item.color} opacity-70 group-hover:opacity-100 transition-opacity`} />
            <div>
              <p className="text-sm font-semibold">{item.label}</p>
              <p className="text-xs text-muted-foreground">{item.desc}</p>
            </div>
          </a>
        ))}
      </div>

      {/* Models Status */}
      <SectionHeader title="Model Pipeline Status" />
      <Card>
        {pipeline.loading && <Spinner />}
        {pipeline.error && <ErrorDisplay message={pipeline.error} />}
        {pipeline.data && (
          <div className="space-y-3">
            <div className="flex gap-4 text-sm">
              <span className="text-emerald font-mono">{pipeline.data.trained} trained</span>
              <span className="text-amber font-mono">{pipeline.data.missing} pending</span>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {pipeline.data.trained_tickers?.map((t: string) => (
                <span key={t} className="rounded-md bg-emerald/10 px-2 py-0.5 text-xs font-mono text-emerald border border-emerald/20">{t}</span>
              ))}
              {pipeline.data.missing_tickers?.map((t: string) => (
                <span key={t} className="rounded-md bg-muted px-2 py-0.5 text-xs font-mono text-muted-foreground">{t}</span>
              ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

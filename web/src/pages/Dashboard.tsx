import { useMemo, useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, PageHeader } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { Brain, TrendingUp, Shield, Zap, Activity, Target } from 'lucide-react'
import { SimpleRecommendationCard } from '@/components/SimpleRecommendationCard'

export default function Dashboard() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const marketStatus = useApi<{ status: string }>('/api/market/status')
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const pipeline = useApi<any>('/api/pipeline/status')
  const monitoring = useApi<any>('/api/monitoring/')
  const recommendation = useApi<any>(`/api/insights/recommend/${ticker}`)
  const enrichment = useApi<any>(`/api/insights/enrichment/${ticker}`)
  const automation = useApi<any>('/api/automation/decisions')
  const paperState = useApi<any>('/api/paper-trading/state')
  const runAutomation = usePostApi<any>('/api/automation/run')

  const quickSummary = useMemo(() => {
    if (!recommendation.data || recommendation.data.error) {
      return 'The recommendation engine is still warming up. Please wait a moment and try again.'
    }
    if (recommendation.data.action === 'BUY') {
      return `A bullish setup is forming for ${ticker}. The system sees improving momentum and a positive trend.`
    }
    if (recommendation.data.action === 'SELL') {
      return `The setup for ${ticker} is turning negative. The system is signaling caution and possible downside.`
    }
    return `The system sees mixed conditions for ${ticker}. A cautious stance is recommended until the signal becomes clearer.`
  }, [recommendation.data, ticker])

  return (
    <div className="space-y-6">
      <PageHeader
        title="Command Center"
        description="Quantitative intelligence terminal for NSE markets"
        badge={marketStatus.data?.status === 'Open' ? 'Live' : 'Warm-up'}
      >
        <div className="flex items-center gap-2 rounded-full border border-border bg-card/70 px-3 py-1.5">
          <div className={`h-2 w-2 rounded-full ${marketStatus.data?.status === 'Open' ? 'bg-emerald animate-pulse' : 'bg-rose'}`} />
          <span className="font-mono text-xs text-muted-foreground">{marketStatus.data?.status || 'Loading...'}</span>
        </div>
      </PageHeader>

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

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold">Automation is wired to both the UI and the CLI</p>
            <p className="text-sm text-muted-foreground">Use the dashboard for instant decisions, or run the same workflow from the terminal with the commands below.</p>
          </div>
          <div className="rounded-lg border border-border bg-background/70 px-3 py-2 font-mono text-xs text-muted-foreground">
            python run_daily.py --paper-trade
          </div>
        </div>
      </Card>

      <SectionHeader title="Simple Decision Guide" />
      <div className="grid gap-4 lg:grid-cols-[1.2fr_0.8fr]">
        <SimpleRecommendationCard data={recommendation.data} loading={recommendation.loading} />
        <Card className="space-y-3">
          <label className="text-sm font-medium">Choose a stock to review</label>
          <select
            value={ticker}
            onChange={(e) => setTicker(e.target.value)}
            className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
          >
            {stocks.data?.stocks?.map((s: string) => (
              <option key={s} value={s}>{s}</option>
            ))}
          </select>
          <p className="text-sm text-muted-foreground">{quickSummary}</p>
          <div className="rounded-xl border border-emerald/20 bg-emerald/10 p-3 text-sm text-emerald">
            Tip: use the backtest page to see how this signal behaved over historical data before taking any action.
          </div>
        </Card>
      </div>

      {/* Automation panel */}
      <SectionHeader title="Automation & Journal" />
      <div className="grid gap-4 lg:grid-cols-[1fr_0.9fr]">
        <Card className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold">Run the daily signal loop</p>
              <p className="text-xs text-muted-foreground">Generate fresh decisions, log them, and keep the journal current.</p>
            </div>
            <button
              onClick={() => runAutomation.post({})}
              className="rounded-lg bg-cyan px-3 py-2 text-sm font-semibold text-background transition hover:opacity-90"
            >
              {runAutomation.loading ? 'Running…' : 'Run Now'}
            </button>
          </div>
          {runAutomation.data?.ok ? (
            <div className="rounded-lg border border-emerald/20 bg-emerald/10 p-3 text-sm text-emerald">
              Daily cycle completed with {runAutomation.data.summary?.decisions?.length || 0} decisions.
            </div>
          ) : runAutomation.error ? (
            <div className="rounded-lg border border-rose/20 bg-rose/10 p-3 text-sm text-rose">{runAutomation.error}</div>
          ) : null}
        </Card>
        <Card className="space-y-3">
          <p className="text-sm font-semibold">Recent journal entries</p>
          {automation.loading && <Spinner />}
          {automation.error && <ErrorDisplay message={automation.error} />}
          <div className="space-y-2">
            {(automation.data?.decisions || []).slice(0, 6).map((entry: any, index: number) => (
              <div key={`${entry.id || index}-${entry.ticker}`} className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2 text-sm">
                <div>
                  <p className="font-mono text-xs text-muted-foreground">{entry.ticker}</p>
                  <p className="font-medium">{entry.action}</p>
                </div>
                <div className="text-right">
                  <p className="text-xs text-muted-foreground">{entry.date}</p>
                  <p className="text-xs text-cyan">{entry.confidence?.toFixed(2) || '—'}</p>
                </div>
              </div>
            ))}
            {(automation.data?.decisions || []).length === 0 && !automation.loading && (
              <p className="text-sm text-muted-foreground">No decisions have been logged yet.</p>
            )}
          </div>
        </Card>
      </div>

      {/* Live context and automation snapshot */}
      <SectionHeader title="Live Market Context" />
      <div className="grid gap-4 lg:grid-cols-[1.1fr_0.9fr]">
        <Card className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold">News & intraday enrichment</p>
              <p className="text-xs text-muted-foreground">Price, momentum, and headline sentiment for {ticker}</p>
            </div>
            <Activity className="h-4 w-4 text-cyan" />
          </div>
          {enrichment.loading && <Spinner />}
          {enrichment.error && <ErrorDisplay message={enrichment.error} />}
          {enrichment.data && !enrichment.data.error && (
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Price</span>
                <span className="font-mono font-semibold">₹{enrichment.data.price?.toLocaleString() || '—'}</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Change</span>
                <span className={`font-mono font-semibold ${enrichment.data.change_pct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                  {enrichment.data.change_pct >= 0 ? '+' : ''}{enrichment.data.change_pct}%
                </span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Sentiment</span>
                <span className="font-medium">{enrichment.data.sentiment_label || 'Neutral'}</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Headlines</span>
                <span className="font-medium">{enrichment.data.headline_count || 0}</span>
              </div>
            </div>
          )}
        </Card>
        <Card className="space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold">Paper trading snapshot</p>
              <p className="text-xs text-muted-foreground">Risk-aware simulated account state</p>
            </div>
            <Shield className="h-4 w-4 text-amber" />
          </div>
          {paperState.loading && <Spinner />}
          {paperState.error && <ErrorDisplay message={paperState.error} />}
          {paperState.data && !paperState.data.error && (
            <div className="space-y-2 text-sm">
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Equity</span>
                <span className="font-mono font-semibold">₹{Math.round(paperState.data.current_equity || 0).toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Cash</span>
                <span className="font-mono font-semibold">₹{Math.round(paperState.data.cash || 0).toLocaleString()}</span>
              </div>
              <div className="flex items-center justify-between rounded-lg border border-border bg-background/70 px-3 py-2">
                <span className="text-muted-foreground">Risk</span>
                <span className={`font-medium ${paperState.data.risk_status?.halted ? 'text-rose' : 'text-emerald'}`}>
                  {paperState.data.risk_status?.halted ? 'Halted' : 'Active'}
                </span>
              </div>
            </div>
          )}
        </Card>
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

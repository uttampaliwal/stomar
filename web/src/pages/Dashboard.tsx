import { useMemo, useState } from 'react'
import { Card, Stat, SectionHeader, Spinner } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { Brain, TrendingUp, Shield, Zap, Activity, Target } from 'lucide-react'
import { SimpleRecommendationCard } from '@/components/SimpleRecommendationCard'
import {
  StatusPill,
  RiskGauge,
  ConsensusCard,
  MLConsensusCard,
  PortfolioHealthCard,
} from '@/components/TerminalComponents'
import type {
  PipelineStatusResponse,
  MonitoringResponse,
  RecommendationResponse,
  EnrichmentResponse,
  AutomationDecisionsResponse,
  AutomationRunResponse,
  PaperPositionSummary,
  ConsensusResponse,
} from '../lib/api-types'

export default function Dashboard() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const marketStatus = useApi<{ status: string }>('/api/market/status')
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')
  const pipeline = useApi<PipelineStatusResponse>('/api/pipeline/status')
  const monitoring = useApi<MonitoringResponse>('/api/monitoring/')
  const recommendation = useApi<RecommendationResponse>(`/api/insights/recommend/${ticker}`)
  const enrichment = useApi<EnrichmentResponse>(`/api/insights/enrichment/${ticker}`)
  const automation = useApi<AutomationDecisionsResponse>('/api/automation/decisions')
  const paperState = useApi<PaperPositionSummary>('/api/paper-trading/state')
  const consensus = useApi<ConsensusResponse>('/api/consensus/')
  const runAutomation = usePostApi<AutomationRunResponse>('/api/automation/run')

  const quickSummary = useMemo(() => {
    if (!recommendation.data || 'error' in recommendation.data) {
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

  const modelVotes = useMemo(() => {
    const pred = recommendation.data
    if (!pred || !('details' in pred)) return []
    const d = pred.details as Record<string, unknown>
    return [
      { name: 'LSTM', direction: d.lstm_dir === 1 ? 'UP' as const : 'DN' as const, confidence: (d.lstm_prob as number) || 0 },
      { name: 'GRU', direction: d.gru_dir === 1 ? 'UP' as const : 'DN' as const, confidence: (d.gru_prob as number) || 0 },
      { name: 'Transformer', direction: d.transformer_dir === 1 ? 'UP' as const : 'DN' as const, confidence: (d.transformer_prob as number) || 0 },
      { name: 'XGBoost', direction: d.xgb_dir === 1 ? 'UP' as const : 'DN' as const, confidence: (d.xgb_prob_up as number) || 0 },
      { name: 'LightGBM', direction: d.lgb_dir === 1 ? 'UP' as const : 'DN' as const, confidence: (d.lgb_prob_up as number) || 0 },
    ]
  }, [recommendation.data])

  const riskScore = useMemo(() => {
    const alerts = monitoring.data?.alerts
    if (!alerts) return 50
    const critical = alerts.critical?.length || 0
    const warning = alerts.warning?.length || 0
    return Math.min(100, critical * 30 + warning * 10)
  }, [monitoring.data])

  const marketIsOpen = marketStatus.data?.status === 'Open'

  return (
    <div className="space-y-4 grid-lines min-h-screen">
      {/* Terminal Status Bar */}
      <div className="flex flex-wrap items-center gap-2 py-1">
        <StatusPill
          label="LIVE FEED"
          value={marketIsOpen ? 'ONLINE' : 'OFFLINE'}
          status={marketIsOpen ? 'active' : 'offline'}
        />
        <StatusPill
          label="RISK ENGINE"
          value={riskScore < 40 ? 'PASSED' : riskScore < 70 ? 'CAUTION' : 'HALT'}
          status={riskScore < 40 ? 'online' : riskScore < 70 ? 'warning' : 'offline'}
        />
        <StatusPill
          label="REGIME"
          value="BULL MARKET"
          status="online"
        />
        <StatusPill
          label="MODELS"
          value={`${pipeline.data?.trained || 0}/${pipeline.data?.total || 20}`}
          status={(pipeline.data?.trained || 0) > 10 ? 'online' : 'warning'}
        />
        <StatusPill
          label="SYSTEM"
          value="ACTIVE"
          status="online"
        />
      </div>

      {/* Dense Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
        <Stat
          label="Stocks"
          value={pipeline.data?.total || '—'}
          sub={`${pipeline.data?.trained || 0} trained`}
          trend="up"
          className="!p-2"
        />
        <Stat
          label="Models"
          value={pipeline.data?.trained || '—'}
          sub={`of ${pipeline.data?.total || 20}`}
          trend={(pipeline.data?.trained || 0) > 0 ? 'up' : 'neutral'}
          className="!p-2"
        />
        <Stat
          label="Alerts"
          value={monitoring.data?.alerts?.critical?.length || 0}
          sub={monitoring.data?.alerts?.warning?.length ? `${monitoring.data.alerts.warning.length} warn` : 'Clear'}
          trend={(monitoring.data?.alerts?.critical?.length || 0) > 0 ? 'down' : 'up'}
          className="!p-2"
        />
        <Stat
          label="BUY"
          value={consensus.data?.buy_count || 0}
          sub="signals"
          trend="up"
          className="!p-2"
        />
        <Stat
          label="SELL"
          value={consensus.data?.sell_count || 0}
          sub="signals"
          trend="down"
          className="!p-2"
        />
        <Stat
          label="HOLD"
          value={consensus.data?.hold_count || 0}
          sub="signals"
          trend="neutral"
          className="!p-2"
        />
      </div>

      {/* Main 3-Column Dashboard Grid */}
      <div className="grid gap-4 lg:grid-cols-[1fr_1.2fr_0.8fr]">
        {/* Left Column: Decision + ML Consensus */}
        <div className="space-y-4">
          <SectionHeader title="Signal Analysis" />
          <SimpleRecommendationCard data={recommendation.data ?? undefined} loading={recommendation.loading} />

          {/* Stock selector */}
          <Card className="!p-3">
            <label className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground mb-1 block">Target Stock</label>
            <select
              value={ticker}
              onChange={(e) => setTicker(e.target.value)}
              className="w-full rounded border border-border bg-background/70 px-2 py-1.5 text-xs font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
            >
              {stocks.data?.stocks?.map((s: string) => (
                <option key={s} value={s}>{s}</option>
              ))}
            </select>
            <p className="mt-2 text-[0.65rem] text-muted-foreground leading-relaxed">{quickSummary}</p>
          </Card>

          {modelVotes.length > 0 && (
            <MLConsensusCard
              modelVotes={modelVotes}
              ensembleDirection={recommendation.data?.action || 'HOLD'}
              ensembleConfidence={recommendation.data?.confidence || 0}
            />
          )}
        </div>

        {/* Center Column: Quick Access + Consensus Grid */}
        <div className="space-y-4">
          <SectionHeader title="Quick Access" />
          <div className="grid grid-cols-3 gap-2">
            {[
              { icon: Brain, label: 'Predictions', desc: 'ML signals', path: '/predictions', color: 'text-violet' },
              { icon: Target, label: 'Consensus', desc: '14-signal', path: '/consensus', color: 'text-cyan' },
              { icon: Shield, label: 'Risk', desc: 'VaR, Kelly', path: '/risk', color: 'text-rose' },
              { icon: Activity, label: 'Market Pulse', desc: 'FII/DII', path: '/market-pulse', color: 'text-emerald' },
              { icon: TrendingUp, label: 'Optimizer', desc: 'Allocation', path: '/optimizer', color: 'text-amber' },
              { icon: Zap, label: 'Paper', desc: 'Simulate', path: '/paper-trading', color: 'text-cyan' },
            ].map((item) => (
              <a
                key={item.path}
                href={item.path}
                className="group flex flex-col items-center gap-1.5 rounded-lg border border-border bg-card/40 p-3 transition-all hover:border-cyan/30 hover:bg-accent/50 text-center"
              >
                <item.icon className={`h-5 w-5 ${item.color} opacity-70 group-hover:opacity-100 transition-opacity`} />
                <p className="text-xs font-semibold">{item.label}</p>
                <p className="text-[0.55rem] text-muted-foreground">{item.desc}</p>
              </a>
            ))}
          </div>

          {/* Consensus Grid */}
          <SectionHeader title="Consensus Signals" />
          <div className="space-y-1 max-h-64 overflow-y-auto">
            {consensus.data?.results?.slice(0, 10).map((r) => (
              <ConsensusCard
                key={r.ticker}
                ticker={r.ticker}
                signal={r.consensus}
                confidence={r.ensemble_confidence}
                regime={r.regime}
              />
            ))}
            {consensus.loading && <Spinner size="sm" />}
          </div>
        </div>

        {/* Right Column: Risk + Portfolio + Automation */}
        <div className="space-y-4">
          <SectionHeader title="Risk & Portfolio" />

          {/* Risk Gauges */}
          <Card className="!p-3 space-y-3">
            <RiskGauge value={riskScore} label="System Risk" size="sm" />
            <RiskGauge value={35} label="Portfolio VaR" size="sm" />
            <RiskGauge value={20} label="Drawdown" size="sm" />
          </Card>

          {/* Portfolio Health */}
          {paperState.data && !('error' in paperState.data) && (
            <PortfolioHealthCard
              equity={paperState.data.current_equity || 0}
              cash={paperState.data.cash || 0}
              pnl={(paperState.data.current_equity || 0) - (paperState.data.initial_capital || 100000)}
              riskStatus={paperState.data.risk_status?.halted ? 'HALTED' : 'Active'}
            />
          )}

          {/* Enrichment Quick View */}
          {enrichment.data && !('error' in enrichment.data) && (
            <Card className="!p-3">
              <div className="flex items-center justify-between mb-2">
                <span className="font-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">{ticker}</span>
                <Activity className="h-3 w-3 text-cyan" />
              </div>
              <div className="grid grid-cols-2 gap-2 text-[0.65rem] font-mono">
                <div>
                  <span className="text-muted-foreground">Price</span>
                  <p className="font-bold">₹{enrichment.data.price?.toLocaleString() || '—'}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Change</span>
                  <p className={`font-bold ${enrichment.data.change_pct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                    {enrichment.data.change_pct >= 0 ? '+' : ''}{enrichment.data.change_pct}%
                  </p>
                </div>
                <div>
                  <span className="text-muted-foreground">Sentiment</span>
                  <p className="font-bold">{enrichment.data.sentiment_label || 'Neutral'}</p>
                </div>
                <div>
                  <span className="text-muted-foreground">Headlines</span>
                  <p className="font-bold">{enrichment.data.headline_count || 0}</p>
                </div>
              </div>
            </Card>
          )}

          {/* Automation */}
          <Card className="!p-3">
            <div className="flex items-center justify-between mb-2">
              <span className="font-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">Automation</span>
              <button
                onClick={() => runAutomation.post({})}
                className="rounded bg-cyan px-2 py-1 text-[0.6rem] font-mono font-bold text-black transition hover:opacity-90"
              >
                {runAutomation.loading ? '...' : 'RUN'}
              </button>
            </div>
            {runAutomation.data?.ok && (
              <p className="text-[0.6rem] text-emerald font-mono">
                {runAutomation.data.summary?.decisions?.length || 0} decisions logged
              </p>
            )}
            <div className="space-y-1 mt-1">
              {(automation.data?.decisions || []).slice(0, 4).map((entry, index) => (
                <div key={`${entry.id || index}-${entry.ticker}`} className="flex items-center justify-between text-[0.6rem] font-mono border-b border-border/50 pb-1">
                  <span className="font-bold">{entry.ticker.replace('.NS', '')}</span>
                  <span className={entry.action === 'BUY' ? 'text-emerald' : entry.action === 'SELL' ? 'text-rose' : 'text-amber'}>
                    {entry.action}
                  </span>
                  <span className="text-muted-foreground">{entry.confidence?.toFixed(2) || '—'}</span>
                </div>
              ))}
            </div>
          </Card>
        </div>
      </div>

      {/* Model Pipeline */}
      <SectionHeader title="Model Pipeline" />
      <Card className="!p-3">
        {pipeline.loading && <Spinner size="sm" />}
        {pipeline.data && (
          <div className="space-y-2">
            <div className="flex gap-3 text-xs font-mono">
              <span className="text-emerald">{pipeline.data.trained} trained</span>
              <span className="text-amber">{pipeline.data.missing} pending</span>
              <span className="text-muted-foreground">|</span>
              <span className="text-muted-foreground">{pipeline.data.total} total</span>
            </div>
            <div className="flex flex-wrap gap-1">
              {pipeline.data.trained_tickers?.map((t: string) => (
                <span key={t} className="rounded bg-emerald/10 px-1.5 py-0.5 text-[0.6rem] font-mono text-emerald border border-emerald/20">{t.replace('.NS', '')}</span>
              ))}
              {pipeline.data.missing_tickers?.map((t: string) => (
                <span key={t} className="rounded bg-muted/30 px-1.5 py-0.5 text-[0.6rem] font-mono text-muted-foreground">{t.replace('.NS', '')}</span>
              ))}
            </div>
          </div>
        )}
      </Card>
    </div>
  )
}

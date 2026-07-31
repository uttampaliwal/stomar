import { useState } from 'react'
import { Card, Stat, SectionHeader, PageHeader, Badge } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'
import { AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Sparkles } from 'lucide-react'
import type { WealthStrategiesResponse, MonteCarloResponse, AdvisorResponse } from '../lib/api-types'

export default function WealthGoals() {
  const { data: strategies } = useApi<WealthStrategiesResponse>('/api/wealth/strategies')
  const { post: runSimulation, loading: simulating, data: simResult } = usePostApi<MonteCarloResponse>('/api/wealth/monte-carlo')
  const { post: getAdvisor, loading: advising, data: advisorResult } = usePostApi<AdvisorResponse>('/api/wealth/wealth-advisor')

  const [form, setForm] = useState({
    target_corpus: 10000000,
    current_capital: 500000,
    monthly_sip: 25000,
    horizon_years: 10,
    expected_return: 0.12,
    volatility: 0.18,
    inflation_rate: 0.06,
  })

  const handleSimulate = async () => {
    await runSimulation(form)
  }

  const handleAdvisor = async () => {
    await getAdvisor({
      target_corpus: form.target_corpus,
      current_capital: form.current_capital,
      monthly_sip: form.monthly_sip,
      horizon_years: form.horizon_years,
      expected_return: form.expected_return,
    })
  }

  const chartData = simResult?.percentiles?.labels?.map((label: string, i: number) => ({
    name: label,
    p10: simResult?.percentiles?.p10[i],
    p50: simResult?.percentiles?.p50[i],
    p90: simResult?.percentiles?.p90[i],
  })) || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Wealth & Goal Engine"
        description="Plan retirement corpus, run Monte Carlo simulations, and explore proven quant strategies"
        badge="Planning"
      />

      {/* Goal Planner Form */}
      <Card className="space-y-4">
        <SectionHeader title="Financial Goal Planner" />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Target Corpus (₹)</label>
            <input
              type="number"
              value={form.target_corpus}
              onChange={(e) => setForm({ ...form, target_corpus: parseInt(e.target.value) || 0 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Current Capital (₹)</label>
            <input
              type="number"
              value={form.current_capital}
              onChange={(e) => setForm({ ...form, current_capital: parseInt(e.target.value) || 0 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Monthly SIP (₹)</label>
            <input
              type="number"
              value={form.monthly_sip}
              onChange={(e) => setForm({ ...form, monthly_sip: parseInt(e.target.value) || 0 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Horizon (Years)</label>
            <input
              type="number"
              value={form.horizon_years}
              onChange={(e) => setForm({ ...form, horizon_years: parseInt(e.target.value) || 1 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Expected Return (%)</label>
            <input
              type="number"
              step="0.01"
              value={(form.expected_return * 100).toFixed(0)}
              onChange={(e) => setForm({ ...form, expected_return: parseFloat(e.target.value) / 100 || 0.12 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Volatility (%)</label>
            <input
              type="number"
              step="0.01"
              value={(form.volatility * 100).toFixed(0)}
              onChange={(e) => setForm({ ...form, volatility: parseFloat(e.target.value) / 100 || 0.18 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
          <div>
            <label className="text-xs text-muted-foreground block mb-1">Inflation (%)</label>
            <input
              type="number"
              step="0.01"
              value={(form.inflation_rate * 100).toFixed(0)}
              onChange={(e) => setForm({ ...form, inflation_rate: parseFloat(e.target.value) / 100 || 0.06 })}
              className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
            />
          </div>
        </div>
        <button
          onClick={handleSimulate}
          disabled={simulating}
          className="rounded-lg bg-cyan px-6 py-2 text-sm font-semibold text-black hover:bg-cyan/80 transition-colors disabled:opacity-50"
        >
          {simulating ? 'Simulating...' : 'Run 500-Run Monte Carlo'}
        </button>
      </Card>

      {/* Simulation Results */}
      {simResult && !('error' in simResult) && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat
              label="Probability of Success"
              value={`${simResult.probability}%`}
              sub={`Target: ${formatCurrency(simResult.target)}`}
              trend={simResult.probability >= 70 ? 'up' : simResult.probability >= 40 ? 'neutral' : 'down'}
            />
            <Stat
              label="Median Corpus"
              value={formatCurrency(simResult.median_corpus)}
              sub="50th percentile"
              trend="up"
            />
            <Stat
              label="Real Corpus (Inflation-Adj)"
              value={formatCurrency(simResult.real_corpus)}
              sub={`${(form.inflation_rate * 100).toFixed(0)}% inflation adjusted`}
              trend="neutral"
            />
            <Stat
              label="Monthly SIP"
              value={formatCurrency(form.monthly_sip)}
              sub={`${form.horizon_years} year horizon`}
              trend="neutral"
            />
          </div>

          {/* Percentile Chart */}
          {chartData.length > 0 && (
            <Card>
              <SectionHeader title="Wealth Projection — Percentile Bands" />
              <div className="h-72">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                    <defs>
                      <linearGradient id="p90fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="p50fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.4} />
                        <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }} />
                    <YAxis
                      tick={{ fontSize: 11, fill: 'hsl(var(--muted-foreground))' }}
                      tickFormatter={(v: number) => `₹${(v / 100000).toFixed(0)}L`}
                    />
                    <Tooltip
                      formatter={(value: number) => formatCurrency(value)}
                      contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8 }}
                    />
                    <ReferenceLine
                      y={simResult.target}
                      stroke="#f43f5e"
                      strokeDasharray="6 3"
                      label={{ value: 'Target', fill: '#f43f5e', fontSize: 11 }}
                    />
                    <Area type="monotone" dataKey="p90" stroke="#22d3ee" fill="url(#p90fill)" strokeWidth={1.5} name="90th %" />
                    <Area type="monotone" dataKey="p50" stroke="#10b981" fill="url(#p50fill)" strokeWidth={2} name="Median" />
                    <Area type="monotone" dataKey="p10" stroke="#a78bfa" fill="transparent" strokeWidth={1.5} strokeDasharray="4 2" name="10th %" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex gap-4 mt-2 text-xs text-muted-foreground">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-cyan inline-block" /> 90th percentile (optimistic)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald inline-block" /> Median (expected)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-violet inline-block border-dashed" /> 10th percentile (conservative)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-rose inline-block border-dashed" /> Target</span>
              </div>
            </Card>
          )}
        </>
      )}

      {/* Wealth Strategies */}
      {strategies?.strategies && (
        <>
          <SectionHeader title="Proven Quant Strategies" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {strategies.strategies.map((s, i: number) => (
              <Card key={i} className="space-y-2">
                <div className="flex items-center justify-between">
                  <p className="text-sm font-semibold">{s.name}</p>
                  <Badge variant="outline">{s.complexity}</Badge>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed">{s.description}</p>
                <div className="flex gap-4 text-xs">
                  <span>CAGR: <span className="font-mono font-semibold text-emerald">{s.cagr}</span></span>
                  <span>Max DD: <span className="font-mono font-semibold text-rose">{s.max_drawdown}</span></span>
                  <span>Sharpe: <span className="font-mono font-semibold text-cyan">{s.sharpe}</span></span>
                </div>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* AI Wealth Advisor */}
      <Card className="space-y-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-violet" />
            <p className="text-sm font-semibold">AI Wealth Advisor</p>
          </div>
          <button
            onClick={handleAdvisor}
            disabled={advising}
            className="rounded-lg bg-violet/10 text-violet border border-violet/20 px-3 py-1.5 text-xs font-semibold hover:bg-violet/20 transition-colors disabled:opacity-50"
          >
            {advising ? 'Analyzing...' : 'Get AI Audit'}
          </button>
        </div>
        {advisorResult?.advisor && (
          <div className="space-y-3 text-sm">
            {advisorResult.source && (
              <Badge variant="outline">{advisorResult.source}</Badge>
            )}
            <p className="text-muted-foreground leading-relaxed">{advisorResult.advisor.executive_summary}</p>
            {advisorResult.advisor.asset_allocation && (
              <div className="flex gap-3 text-xs">
                {Object.entries(advisorResult.advisor.asset_allocation).map(([k, v]) => (
                  <span key={k} className="font-mono">
                    {k.replace('_pct', '').toUpperCase()}: <span className="font-semibold">{v as number}%</span>
                  </span>
                ))}
              </div>
            )}
            {advisorResult.advisor.action_items && (
              <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
                {advisorResult.advisor.action_items.map((item: string, i: number) => (
                  <li key={i}>{item}</li>
                ))}
              </ul>
            )}
            {advisorResult.advisor.risk_warnings && (
              <div className="rounded-lg bg-amber/5 border border-amber/20 p-2 text-xs text-amber space-y-1">
                {advisorResult.advisor.risk_warnings.map((w: string, i: number) => (
                  <p key={i}>⚠ {w}</p>
                ))}
              </div>
            )}
            {advisorResult.note && (
              <p className="text-[10px] text-muted-foreground italic">{advisorResult.note}</p>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}

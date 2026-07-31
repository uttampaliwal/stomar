import { useState, useCallback } from 'react'
import { Card, Stat, SectionHeader, PageHeader, Badge, Spinner } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ReferenceLine, ComposedChart, Bar,
} from 'recharts'
import { Sparkles, TrendingUp, AlertTriangle, Target, Coins, Zap } from 'lucide-react'
import type { WealthStrategiesResponse, MonteCarloResponse, AdvisorResponse } from '../lib/api-types'

function Slider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  format,
  unit = '',
}: {
  label: string
  value: number
  onChange: (v: number) => void
  min: number
  max: number
  step: number
  format?: (v: number) => string
  unit?: string
}) {
  const display = format ? format(value) : `${value.toLocaleString('en-IN')}${unit}`
  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="text-[0.65rem] font-mono uppercase tracking-wider text-muted-foreground">{label}</label>
        <span className="font-mono text-xs font-bold text-foreground tabular-nums">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(parseFloat(e.target.value))}
        className="w-full h-1.5 rounded-full appearance-none cursor-pointer bg-muted/30
          [&::-webkit-slider-thumb]:appearance-none [&::-webkit-slider-thumb]:h-3.5 [&::-webkit-slider-thumb]:w-3.5
          [&::-webkit-slider-thumb]:rounded-full [&::-webkit-slider-thumb]:bg-cyan [&::-webkit-slider-thumb]:cursor-pointer
          [&::-webkit-slider-thumb]:shadow-[0_0_6px_rgba(34,211,238,0.5)]
          [&::-moz-range-thumb]:h-3.5 [&::-moz-range-thumb]:w-3.5 [&::-moz-range-thumb]:rounded-full
          [&::-moz-range-thumb]:bg-cyan [&::-moz-range-thumb]:border-0 [&::-moz-range-thumb]:cursor-pointer"
      />
      <div className="flex justify-between text-[0.55rem] font-mono text-muted-foreground/50">
        <span>{format ? format(min) : min.toLocaleString('en-IN')}{unit}</span>
        <span>{format ? format(max) : max.toLocaleString('en-IN')}{unit}</span>
      </div>
    </div>
  )
}

const fmtINR = (v: number) => {
  if (v >= 1e7) return `₹${(v / 1e7).toFixed(2)} Cr`
  if (v >= 1e5) return `₹${(v / 1e5).toFixed(1)} L`
  return `₹${v.toLocaleString('en-IN', { maximumFractionDigits: 0 })}`
}

export default function WealthGoals() {
  const { data: strategies } = useApi<WealthStrategiesResponse>('/api/wealth/strategies')
  const { post: runSimulation, loading: simulating, data: simResult } = usePostApi<MonteCarloResponse>('/api/wealth/monte-carlo')
  const { post: getAdvisor, loading: advising, data: advisorResult } = usePostApi<AdvisorResponse>('/api/wealth/wealth-advisor')

  const [form, setForm] = useState({
    target_corpus: 10000000,
    current_capital: 500000,
    monthly_sip: 25000,
    step_up_pct: 0.10,
    horizon_years: 10,
    expected_return: 0.12,
    volatility: 0.18,
    inflation_rate: 0.06,
    num_simulations: 1000,
    t_df: 5,
  })

  const update = useCallback((key: string, val: number) => {
    setForm((prev) => ({ ...prev, [key]: val }))
  }, [])

  const handleSimulate = async () => {
    await runSimulation(form)
  }

  const handleAdvisor = async () => {
    await getAdvisor({
      target_corpus: form.target_corpus,
      current_capital: form.current_capital,
      monthly_sip: form.monthly_sip,
      step_up_pct: form.step_up_pct,
      horizon_years: form.horizon_years,
      expected_return: form.expected_return,
    })
  }

  const chartData = simResult?.percentiles?.labels?.map((label: string, i: number) => ({
    name: label,
    p5: simResult?.percentiles?.p5[i],
    p10: simResult?.percentiles?.p10[i],
    p25: simResult?.percentiles?.p25[i],
    p50: simResult?.percentiles?.p50[i],
    p75: simResult?.percentiles?.p75[i],
    p90: simResult?.percentiles?.p90[i],
    p95: simResult?.percentiles?.p95[i],
  })) || []

  // Distribution histogram data from percentiles
  const distData = simResult ? [
    { range: 'Worst 5%', value: simResult.worst_5pct, color: '#f43f5e' },
    { range: '10th %', value: simResult.percentiles.p10[simResult.percentiles.p10.length - 1], color: '#a78bfa' },
    { range: '25th %', value: simResult.percentiles.p25[simResult.percentiles.p25.length - 1], color: '#f59e0b' },
    { range: 'Median', value: simResult.median_corpus, color: '#10b981' },
    { range: '75th %', value: simResult.percentiles.p75[simResult.percentiles.p75.length - 1], color: '#22d3ee' },
    { range: '90th %', value: simResult.percentiles.p90[simResult.percentiles.p90.length - 1], color: '#8b5cf6' },
    { range: 'Best 5%', value: simResult.best_5pct, color: '#10b981' },
  ] : []

  return (
    <div className="space-y-4 grid-lines min-h-screen">
      <PageHeader
        title="Wealth & Goal Engine"
        description="Monte Carlo simulation with fat-tailed distributions, step-up SIP, and inflation-adjusted projections"
        badge="Planning"
      >
        <div className="flex items-center gap-2">
          <span className="font-mono text-[0.6rem] text-muted-foreground">
            {form.num_simulations.toLocaleString()} paths • Student's t(df={form.t_df})
          </span>
        </div>
      </PageHeader>

      {/* Interactive Parameter Panel */}
      <div className="grid gap-4 lg:grid-cols-[1fr_320px]">
        {/* Left: Sliders */}
        <Card className="!p-4">
          <SectionHeader title="Goal Parameters" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-6 gap-y-4">
            <Slider
              label="Target Corpus"
              value={form.target_corpus}
              onChange={(v) => update('target_corpus', v)}
              min={100000}
              max={100000000}
              step={500000}
              format={fmtINR}
            />
            <Slider
              label="Current Capital"
              value={form.current_capital}
              onChange={(v) => update('current_capital', v)}
              min={0}
              max={50000000}
              step={100000}
              format={fmtINR}
            />
            <Slider
              label="Monthly SIP"
              value={form.monthly_sip}
              onChange={(v) => update('monthly_sip', v)}
              min={1000}
              max={500000}
              step={1000}
              format={fmtINR}
            />
            <Slider
              label="Step-Up SIP (Annual)"
              value={form.step_up_pct}
              onChange={(v) => update('step_up_pct', v)}
              min={0}
              max={0.30}
              step={0.01}
              format={(v) => `${(v * 100).toFixed(0)}%`}
            />
            <Slider
              label="Investment Horizon"
              value={form.horizon_years}
              onChange={(v) => update('horizon_years', v)}
              min={1}
              max={40}
              step={1}
              format={(v) => `${v} years`}
            />
            <Slider
              label="Expected Return"
              value={form.expected_return}
              onChange={(v) => update('expected_return', v)}
              min={0.04}
              max={0.25}
              step={0.005}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
            <Slider
              label="Volatility"
              value={form.volatility}
              onChange={(v) => update('volatility', v)}
              min={0.05}
              max={0.40}
              step={0.005}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
            <Slider
              label="Inflation Rate"
              value={form.inflation_rate}
              onChange={(v) => update('inflation_rate', v)}
              min={0.02}
              max={0.12}
              step={0.005}
              format={(v) => `${(v * 100).toFixed(1)}%`}
            />
          </div>
          <div className="flex items-center gap-3 mt-4 pt-3 border-t border-border">
            <Slider
              label="Simulations"
              value={form.num_simulations}
              onChange={(v) => update('num_simulations', Math.round(v / 100) * 100)}
              min={1000}
              max={10000}
              step={500}
              format={(v) => v.toLocaleString()}
            />
            <Slider
              label="Fat Tails (df)"
              value={form.t_df}
              onChange={(v) => update('t_df', Math.round(v))}
              min={3}
              max={30}
              step={1}
              format={(v) => `df=${v}`}
            />
          </div>
          <button
            onClick={handleSimulate}
            disabled={simulating}
            className="mt-4 rounded-lg bg-cyan px-6 py-2.5 text-sm font-mono font-bold text-black hover:bg-cyan/80 transition-colors disabled:opacity-50 flex items-center gap-2"
          >
            {simulating ? (
              <>
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-black border-t-transparent" />
                Simulating...
              </>
            ) : (
              <>
                <Zap className="h-3.5 w-3.5" />
                Run Monte Carlo
              </>
            )}
          </button>
        </Card>

        {/* Right: Key Metrics */}
        <div className="space-y-3">
          {simResult && !('error' in simResult) ? (
            <>
              {/* Probability Gauge */}
              <Card className="!p-4 text-center">
                <p className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground mb-1">Goal Probability</p>
                <p className={`font-mono text-4xl font-bold ${simResult.probability >= 70 ? 'text-emerald' : simResult.probability >= 40 ? 'text-amber' : 'text-rose'}`}>
                  {simResult.probability}%
                </p>
                <p className="text-xs text-muted-foreground mt-1">{formatCurrency(simResult.target)} target</p>
                {/* Progress bar */}
                <div className="mt-3 h-2 rounded-full bg-muted/30 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-700 ${simResult.probability >= 70 ? 'bg-emerald' : simResult.probability >= 40 ? 'bg-amber' : 'bg-rose'}`}
                    style={{ width: `${Math.min(simResult.probability, 100)}%` }}
                  />
                </div>
              </Card>

              {/* Shortfall Risk */}
              <Card className="!p-3">
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className="h-3.5 w-3.5 text-amber" />
                  <span className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground">Risk Analysis</span>
                </div>
                <div className="space-y-2 text-[0.65rem] font-mono">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Shortfall Risk</span>
                    <span className={`font-bold ${simResult.shortfall_risk > 30 ? 'text-rose' : 'text-amber'}`}>{simResult.shortfall_risk}%</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">VaR (95%)</span>
                    <span className="text-rose font-bold">{fmtINR(simResult.var_95)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">CVaR (95%)</span>
                    <span className="text-rose font-bold">{fmtINR(simResult.cvar_95)}</span>
                  </div>
                </div>
                {simResult.sip_topup > 0 && (
                  <div className="mt-2 rounded border border-amber/20 bg-amber/5 p-2">
                    <p className="text-[0.6rem] font-mono text-amber">
                      Top-up suggested: <span className="font-bold">+{fmtINR(simResult.sip_topup)}/mo</span> for 80% goal probability
                    </p>
                  </div>
                )}
              </Card>

              {/* Inflation-Adjusted */}
              <Card className="!p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Coins className="h-3.5 w-3.5 text-cyan" />
                  <span className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground">Real Wealth</span>
                </div>
                <div className="space-y-2 text-[0.65rem] font-mono">
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">Median (Today's ₹)</span>
                    <span className="text-emerald font-bold">{fmtINR(simResult.real_corpus)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">10th %ile (Today's ₹)</span>
                    <span className="text-rose font-bold">{fmtINR(simResult.real_p10)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-muted-foreground">90th %ile (Today's ₹)</span>
                    <span className="text-emerald font-bold">{fmtINR(simResult.real_p90)}</span>
                  </div>
                </div>
              </Card>

              {/* Recommended SIP */}
              <Card className="!p-3">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="h-3.5 w-3.5 text-violet" />
                  <span className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground">Recommended SIP</span>
                </div>
                <p className="font-mono text-lg font-bold text-violet">{fmtINR(simResult.recommended_sip)}/mo</p>
                <p className="text-[0.55rem] text-muted-foreground mt-0.5">for higher confidence target</p>
              </Card>
            </>
          ) : (
            <Card className="!p-6 text-center">
              <TrendingUp className="h-8 w-8 text-muted-foreground/30 mx-auto mb-2" />
              <p className="text-xs text-muted-foreground font-mono">Adjust parameters and run simulation</p>
            </Card>
          )}
        </div>
      </div>

      {/* Simulation Results Charts */}
      {simResult && !('error' in simResult) && (
        <>
          {/* Stats Row */}
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
            <Stat label="Probability" value={`${simResult.probability}%`} sub="of reaching target" trend={simResult.probability >= 70 ? 'up' : simResult.probability >= 40 ? 'neutral' : 'down'} className="!p-2" />
            <Stat label="Median Corpus" value={fmtINR(simResult.median_corpus)} sub="50th percentile" trend="up" className="!p-2" />
            <Stat label="Real Corpus" value={fmtINR(simResult.real_corpus)} sub="inflation-adjusted" trend="neutral" className="!p-2" />
            <Stat label="Worst 5%" value={fmtINR(simResult.worst_5pct)} sub="downside" trend="down" className="!p-2" />
            <Stat label="Best 5%" value={fmtINR(simResult.best_5pct)} sub="upside" trend="up" className="!p-2" />
            <Stat label="Monthly SIP" value={fmtINR(form.monthly_sip)} sub={`+${(form.step_up_pct * 100).toFixed(0)}% step-up`} trend="neutral" className="!p-2" />
          </div>

          {/* Percentile Band Chart */}
          {chartData.length > 0 && (
            <Card className="!p-4">
              <SectionHeader title="Wealth Projection — Confidence Bands" />
              <div className="h-80">
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                    <defs>
                      <linearGradient id="band90fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                      </linearGradient>
                      <linearGradient id="band50fill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#10b981" stopOpacity={0.35} />
                        <stop offset="100%" stopColor="#10b981" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="name" tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))', fontFamily: "'JetBrains Mono'" }} />
                    <YAxis
                      tick={{ fontSize: 10, fill: 'hsl(var(--muted-foreground))', fontFamily: "'JetBrains Mono'" }}
                      tickFormatter={(v: number) => `₹${(v / 100000).toFixed(0)}L`}
                      width={50}
                    />
                    <Tooltip
                      formatter={(value: number, name: string) => [fmtINR(value), name]}
                      contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11, fontFamily: "'JetBrains Mono'" }}
                    />
                    <ReferenceLine y={simResult.target} stroke="#f43f5e" strokeDasharray="6 3" label={{ value: 'TARGET', fill: '#f43f5e', fontSize: 10, fontFamily: "'JetBrains Mono'" }} />
                    {/* 5th-95th band (outermost) */}
                    <Area type="monotone" dataKey="p95" stroke="transparent" fill="url(#band90fill)" strokeWidth={0} name="95th %" />
                    <Area type="monotone" dataKey="p5" stroke="transparent" fill="transparent" strokeWidth={0} name="5th %" />
                    {/* 10th-90th band */}
                    <Area type="monotone" dataKey="p90" stroke="#22d3ee" fill="url(#band90fill)" strokeWidth={1} strokeDasharray="4 2" name="90th %" />
                    <Area type="monotone" dataKey="p10" stroke="#a78bfa" fill="transparent" strokeWidth={1} strokeDasharray="4 2" name="10th %" />
                    {/* 25th-75th band */}
                    <Area type="monotone" dataKey="p75" stroke="#22d3ee" fill="transparent" strokeWidth={0.5} name="75th %" />
                    <Area type="monotone" dataKey="p25" stroke="#f59e0b" fill="transparent" strokeWidth={0.5} name="25th %" />
                    {/* Median */}
                    <Area type="monotone" dataKey="p50" stroke="#10b981" fill="url(#band50fill)" strokeWidth={2} name="Median" />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <div className="flex flex-wrap gap-3 mt-2 text-[0.6rem] font-mono text-muted-foreground">
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-cyan inline-block" /> 90th (optimistic)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-emerald inline-block" /> Median</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-violet inline-block" /> 10th (conservative)</span>
                <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-rose inline-block" /> Target</span>
              </div>
            </Card>
          )}

          {/* Distribution Histogram */}
          {distData.length > 0 && (
            <Card className="!p-4">
              <SectionHeader title="Terminal Wealth Distribution" />
              <div className="h-48">
                <ResponsiveContainer width="100%" height="100%">
                  <ComposedChart data={distData} margin={{ top: 10, right: 10, bottom: 0, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                    <XAxis dataKey="range" tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))', fontFamily: "'JetBrains Mono'" }} />
                    <YAxis tick={{ fontSize: 9, fill: 'hsl(var(--muted-foreground))', fontFamily: "'JetBrains Mono'" }} tickFormatter={(v: number) => `₹${(v / 100000).toFixed(0)}L`} width={50} />
                    <Tooltip
                      formatter={(value: number) => [fmtINR(value), 'Wealth']}
                      contentStyle={{ background: 'hsl(var(--card))', border: '1px solid hsl(var(--border))', borderRadius: 8, fontSize: 11, fontFamily: "'JetBrains Mono'" }}
                    />
                    <ReferenceLine y={simResult.target} stroke="#f43f5e" strokeDasharray="4 2" />
                    <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                      {distData.map((entry, i) => (
                        <rect key={i} fill={entry.color} />
                      ))}
                    </Bar>
                  </ComposedChart>
                </ResponsiveContainer>
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
              <Card key={i} className="!p-3">
                <div className="flex items-center justify-between mb-1.5">
                  <p className="text-sm font-semibold">{s.name}</p>
                  <Badge variant="outline">{s.complexity}</Badge>
                </div>
                <p className="text-[0.65rem] text-muted-foreground leading-relaxed mb-2">{s.description}</p>
                <div className="flex gap-4 text-[0.65rem] font-mono">
                  <span>CAGR: <span className="font-bold text-emerald">{s.cagr}</span></span>
                  <span>Max DD: <span className="font-bold text-rose">{s.max_drawdown}</span></span>
                  <span>Sharpe: <span className="font-bold text-cyan">{s.sharpe}</span></span>
                </div>
              </Card>
            ))}
          </div>
        </>
      )}

      {/* AI Wealth Advisor */}
      <Card className="!p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <Sparkles className="h-4 w-4 text-violet" />
            <p className="text-sm font-semibold">AI Wealth Advisor</p>
            <Badge variant="info">Gemini</Badge>
          </div>
          <button
            onClick={handleAdvisor}
            disabled={advising}
            className="rounded-lg bg-violet/10 text-violet border border-violet/20 px-3 py-1.5 text-xs font-mono font-bold hover:bg-violet/20 transition-colors disabled:opacity-50 flex items-center gap-1.5"
          >
            {advising ? (
              <>
                <div className="h-3 w-3 animate-spin rounded-full border-2 border-violet border-t-transparent" />
                Analyzing...
              </>
            ) : (
              <>
                <Sparkles className="h-3 w-3" />
                Get AI Audit
              </>
            )}
          </button>
        </div>
        {advisorResult?.advisor && (
          <div className="space-y-3 text-sm">
            {advisorResult.source && <Badge variant="outline">{advisorResult.source}</Badge>}
            <p className="text-muted-foreground leading-relaxed text-xs">{advisorResult.advisor.executive_summary}</p>
            {advisorResult.advisor.asset_allocation && (
              <div className="flex gap-3 text-[0.65rem] font-mono">
                {Object.entries(advisorResult.advisor.asset_allocation).map(([k, v]) => (
                  <span key={k}>
                    {k.replace('_pct', '').toUpperCase()}: <span className="font-bold">{v as number}%</span>
                  </span>
                ))}
              </div>
            )}
            {advisorResult.advisor.recommended_funds && (
              <div>
                <p className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground mb-1">Recommended Funds</p>
                <div className="flex flex-wrap gap-1.5">
                  {advisorResult.advisor.recommended_funds.map((f: string, i: number) => (
                    <span key={i} className="rounded border border-border bg-muted/30 px-2 py-0.5 text-[0.6rem] font-mono">{f}</span>
                  ))}
                </div>
              </div>
            )}
            {advisorResult.advisor.action_items && (
              <div>
                <p className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground mb-1">Action Items</p>
                <ul className="text-xs text-muted-foreground space-y-1 list-disc list-inside">
                  {advisorResult.advisor.action_items.map((item: string, i: number) => (
                    <li key={i}>{item}</li>
                  ))}
                </ul>
              </div>
            )}
            {advisorResult.advisor.risk_warnings && (
              <div className="rounded-lg bg-amber/5 border border-amber/20 p-2 text-xs text-amber space-y-1">
                {advisorResult.advisor.risk_warnings.map((w: string, i: number) => (
                  <p key={i}>⚠ {w}</p>
                ))}
              </div>
            )}
            {advisorResult.note && (
              <p className="text-[0.55rem] text-muted-foreground italic">{advisorResult.note}</p>
            )}
          </div>
        )}
      </Card>
    </div>
  )
}

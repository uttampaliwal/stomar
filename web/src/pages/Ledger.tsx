import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { apiFetch } from '@/lib/api-client'
import {
  ResponsiveContainer, ComposedChart, Line, Bar, XAxis, YAxis,
  Tooltip, Legend, CartesianGrid,
} from 'recharts'
import type {
  LedgerSummaryResponse, LedgerDecisionsResponse, LedgerDecision,
  LedgerBenchmarkResponse, CalibrationResponse, AgreementStats,
} from '../lib/api-types'

function buildBenchmarkData(b: LedgerBenchmarkResponse) {
  if (!b.paper || !b.nifty) return []
  const paperMap = new Map<number, number>()
  b.paper.dates.forEach((d, i) => paperMap.set(Date.parse(d), b.paper!.cumulative[i]))
  const rows: { date: string; paper: number | null; nifty: number | null }[] = []
  b.nifty.dates.forEach((d, i) => {
    const ts = Date.parse(d)
    const nearest = [...paperMap.keys()].filter((k) => k <= ts).sort((a, c) => c - a)[0]
    rows.push({
      date: d.slice(5),
      nifty: Math.round(b.nifty!.cumulative[i] * 10000) / 100,
      paper: nearest !== undefined ? Math.round(paperMap.get(nearest)! * 10000) / 100 : null,
    })
  })
  return rows
}

export default function Ledger() {
  const { data, loading, error, refetch } = useApi<LedgerSummaryResponse>('/api/ledger/summary')
  const [agreeFilter, setAgreeFilter] = useState(false)
  const { data: decisionsData } = useApi<LedgerDecisionsResponse>(
    '/api/ledger/decisions' + (agreeFilter ? '?agreement=all' : '')
  )
  const { data: agreementStats } = useApi<AgreementStats>('/api/ledger/agreement-stats')
  const { data: benchmarkData, loading: benchmarkLoading } = useApi<LedgerBenchmarkResponse>('/api/ledger/benchmark')
  const { data: calibrationData } = useApi<CalibrationResponse>('/api/ledger/calibration')
  const [running, setRunning] = useState(false)
  const [runResult, setRunResult] = useState<string | null>(null)
  const [expandedId, setExpandedId] = useState<number | null>(null)
  const benchmarkRows = benchmarkData && !('error' in benchmarkData) ? buildBenchmarkData(benchmarkData) : []

  const handleRunPipeline = async () => {
    setRunning(true)
    setRunResult(null)
    try {
      const res = await apiFetch('/api/pipeline/run', { method: 'POST' })
      const data = await res.json()
      if (data.status === 'started') {
        setRunResult(`Pipeline started for ${data.tickers?.length || 0} tickers. Monitoring...`)
        // Poll for completion
        const poll = setInterval(async () => {
          const statusRes = await apiFetch('/api/pipeline/run/status')
          const status = await statusRes.json()
          if (status.status === 'done') {
            clearInterval(poll)
            const d = status.result
            setRunResult(
              `Complete: ${d?.decisions?.length || 0} decisions, ` +
              `${d?.trades?.length || 0} trades, ${d?.errors?.length || 0} errors`
            )
            setRunning(false)
            refetch()
          } else if (status.status === 'error') {
            clearInterval(poll)
            setRunResult(`Error: ${status.error}`)
            setRunning(false)
          }
        }, 3000)
      } else if (data.status === 'already_running') {
        setRunResult('Pipeline already running...')
      } else if (data.status === 'no_trained_models') {
        setRunResult('No trained models found. Train models first from Pipeline page.')
        setRunning(false)
      } else {
        setRunResult(JSON.stringify(data))
        setRunning(false)
      }
    } catch (e) {
      setRunResult(`Failed: ${e instanceof Error ? e.message : String(e)}`)
      setRunning(false)
    }
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Trading Ledger"
        description="Decision history, signal accuracy, and P&L tracking"
        badge="Audit"
      >
        <button
          onClick={handleRunPipeline}
          disabled={running}
          className="rounded-lg bg-cyan px-4 py-2 text-sm font-semibold text-black transition hover:opacity-90 disabled:opacity-50"
        >
          {running ? 'Running...' : 'Run Pipeline'}
        </button>
      </PageHeader>

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <p className="text-sm text-muted-foreground">
            Use this ledger to review historical decisions and trace how the automation pipeline behaved over time.
          </p>
          <div className="rounded-lg border border-border bg-background/70 px-3 py-2 font-mono text-xs text-muted-foreground">
            python run_daily.py --backfill --days 252
          </div>
        </div>
      </Card>

      {runResult && (
        <div className="rounded-lg border border-border bg-card/70 p-3 text-sm font-mono text-muted-foreground">{runResult}</div>
      )}

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !('error' in data) && (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Total Decisions" value={data.total_decisions || 0} />
            <Stat label="Total Trades" value={data.total_trades || 0} />
            <Stat label="Accuracy" value={`${((data.accuracy || 0) * 100).toFixed(1)}%`} trend={data.accuracy > 0.5 ? 'up' : 'down'} />
            <Stat label="Trade Accuracy (BUY/SELL)" value={data.trade_accuracy != null ? `${(data.trade_accuracy * 100).toFixed(1)}%` : 'N/A'} trend={(data.trade_accuracy || 0) > 0.5 ? 'up' : 'down'} />
          </div>
          <p className="text-xs text-muted-foreground -mt-2">
            Trade accuracy excludes HOLD decisions (which put no capital at risk) — it is the honest hit rate of actual trades.
          </p>

          {/* Paper vs Nifty benchmark (P3.2) */}
          {benchmarkData && !('error' in benchmarkData) && benchmarkRows.length > 1 ? (
            <>
              <SectionHeader title="Paper Trading vs Nifty 50 (cumulative return %)" />
              <Card>
                <div className="h-64">
                  <ResponsiveContainer width="100%" height="100%">
                    <ComposedChart data={benchmarkRows} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" opacity={0.2} />
                      <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={40} />
                      <YAxis tick={{ fontSize: 11 }} unit="%" />
                      <Tooltip
                        formatter={(value: number) => [value == null ? '-' : `${value}%`]}
                        labelStyle={{ color: '#666' }}
                      />
                      <Legend />
                      <Line type="monotone" dataKey="paper" name="Paper Trading" stroke="#22d3ee" strokeWidth={2} dot={false} connectNulls />
                      <Line type="monotone" dataKey="nifty" name="Nifty 50" stroke="#a78bfa" strokeWidth={2} dot={false} />
                    </ComposedChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-3 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <Stat label="Paper Return" value={`${((benchmarkData.paper_total_return || 0) * 100).toFixed(2)}%`} />
                  <Stat label="Nifty Return" value={`${((benchmarkData.nifty_total_return || 0) * 100).toFixed(2)}%`} />
                  <Stat
                    label="Alpha"
                    value={`${((benchmarkData.alpha || 0) * 100).toFixed(2)}%`}
                    trend={(benchmarkData.alpha || 0) > 0 ? 'up' : 'down'}
                  />
                  <Stat
                    label="Info Ratio"
                    value={benchmarkData.information_ratio != null ? benchmarkData.information_ratio.toFixed(2) : 'N/A'}
                  />
                </div>
                <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                  <Stat label="Paper Max DD" value={`${((benchmarkData.paper_max_drawdown || 0) * 100).toFixed(1)}%`} />
                  <Stat label="Nifty Max DD" value={`${((benchmarkData.nifty_max_drawdown || 0) * 100).toFixed(1)}%`} />
                  <Stat label="Snapshots" value={benchmarkData.n_snapshots || 0} />
                  <Stat label="Nifty Points" value={benchmarkData.nifty_points || 0} />
                </div>
              </Card>
            </>
          ) : benchmarkData && 'error' in benchmarkData ? (
            <Card>
              <p className="text-xs text-muted-foreground">Benchmark unavailable: {benchmarkData.error}</p>
            </Card>
          ) : benchmarkLoading ? null : (
            <Card>
              <p className="text-xs text-muted-foreground">
                Benchmark chart appears once paper trading snapshots exist (P3.2).
              </p>
            </Card>
          )}

          {/* Calibration (P4.4) */}
          {calibrationData && !('error' in calibrationData) && calibrationData.n > 0 && (
            <>
              <SectionHeader title="Confidence Calibration" />
              <Card>
                <p className="mb-2 text-xs text-muted-foreground">
                  When the system says "68% confidence", is it right 68% of the time?
                  ECE (expected calibration error): {calibrationData.ece ?? 'N/A'} — closer to 0 is better.
                </p>
                <div className="space-y-2">
                  {calibrationData.bins?.filter((b) => b.n > 0).map((b) => {
                    const diff = Math.abs((b.empirical_accuracy || 0) - (b.mean_confidence || 0))
                    return (
                      <div key={b.bin} className="flex items-center gap-3">
                        <span className="w-24 text-xs font-mono uppercase text-muted-foreground">{b.bin}</span>
                        <div className="flex-1 h-2 rounded-full bg-muted overflow-hidden">
                          <div
                            className={`h-full rounded-full transition-all ${diff < 0.1 ? 'bg-emerald' : diff < 0.25 ? 'bg-amber' : 'bg-rose'}`}
                            style={{ width: `${(b.empirical_accuracy || 0) * 100}%` }}
                          />
                        </div>
                        <span className="font-mono text-xs w-24 text-right">
                          {((b.empirical_accuracy || 0) * 100).toFixed(0)}% vs {(b.mean_confidence || 0) * 100}%
                        </span>
                        <span className="w-10 text-right text-xs text-muted-foreground">n={b.n}</span>
                      </div>
                    )
                  })}
                  {calibrationData.bins?.every((b) => b.n === 0) && (
                    <p className="text-xs text-muted-foreground">No resolved decisions yet — calibration needs live outcome data.</p>
                  )}
                </div>
              </Card>
            </>
          )}

          {/* Signal Accuracy */}
          {data.signal_accuracy && Object.keys(data.signal_accuracy).length > 0 && (
            <>
              <SectionHeader title="Signal Module Accuracy" />
              <Card>
                <div className="space-y-2">
                  {Object.entries(data.signal_accuracy).map(([module, acc]) => (
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
                  {data.snapshots.map((s, i) => {
                    const val = s.total_value || 0
                    const maxVal = Math.max(...data.snapshots.map((x) => x.total_value || 0))
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

          {/* P4.2: Decision audit trail — agreement filter + signal details */}
          {agreementStats && !('error' in agreementStats) && (
            <Card>
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                <div className="space-y-1">
                  <p className="text-sm font-semibold">Main-Signal Agreement Audit (P4.2)</p>
                  <p className="text-xs text-muted-foreground">
                    When ensemble + sentiment + regime all point the same way, is the system more accurate?
                  </p>
                  <div className="flex gap-4 text-xs font-mono">
                    <span>
                      All decisions: <b>{agreementStats.all.n}</b>{' '}
                      ({agreementStats.all.accuracy != null ? `${(agreementStats.all.accuracy * 100).toFixed(1)}%` : 'N/A'})
                    </span>
                    <span>
                      Agreed: <b>{agreementStats.agreed.n}</b>{' '}
                      ({agreementStats.agreed.accuracy != null ? `${(agreementStats.agreed.accuracy * 100).toFixed(1)}%` : 'N/A'})
                    </span>
                  </div>
                </div>
                <label className="flex items-center gap-2 text-xs font-mono cursor-pointer">
                  <input
                    type="checkbox"
                    checked={agreeFilter}
                    onChange={(e) => setAgreeFilter(e.target.checked)}
                    className="accent-cyan"
                  />
                  Only decisions where main signals agreed
                </label>
              </div>
            </Card>
          )}

          {/* Decisions Table */}
          {decisionsData && decisionsData.decisions.length > 0 ? (
            <>
              <SectionHeader title="Recent Decisions" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border">
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Date</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Ticker</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Action</th>
                      <th className="text-right py-2 px-3 font-mono text-xs text-muted-foreground">Confidence</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Regime</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Source</th>
                      <th className="text-left py-2 px-3 font-mono text-xs text-muted-foreground">Reasoning</th>
                    </tr>
                  </thead>
                  <tbody>
                    {decisionsData.decisions.slice(0, 20).map((d: LedgerDecision) => (
                      <AuditRow key={d.id} d={d} expanded={expandedId === d.id} onToggle={() => setExpandedId(expandedId === d.id ? null : d.id)} />
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          ) : (
            <EmptyState message="No decisions have been logged yet." />
          )}
        </>
      )}
    </div>
  )
}

function AuditRow({ d, expanded, onToggle }: { d: LedgerDecision; expanded: boolean; onToggle: () => void }) {
  const rows: [string, string][] = [
    ['Ensemble direction', d.ensemble_direction == null ? 'N/A' : d.ensemble_direction === 1 ? 'UP' : 'DOWN'],
    ['Ensemble confidence', d.ensemble_confidence != null ? `${(d.ensemble_confidence * 100).toFixed(0)}%` : 'N/A'],
    ['Sentiment', d.sentiment_score != null ? `${d.sentiment_score >= 0 ? '+' : ''}${d.sentiment_score.toFixed(2)}` : 'N/A'],
    ['FII net', d.fii_net != null ? `${d.fii_net.toFixed(2)}` : 'N/A'],
    ['DII net', d.dii_net != null ? `${d.dii_net.toFixed(2)}` : 'N/A'],
    ['PCR (OI)', d.pcr != null ? d.pcr.toFixed(2) : 'N/A'],
    ['Max Pain', d.max_pain != null ? d.max_pain.toFixed(0) : 'N/A'],
    ['MTF signal', d.mtf_signal != null ? d.mtf_signal.toFixed(2) : 'N/A'],
    ['Regime', d.regime ?? 'N/A'],
    ['VaR 95', d.var_95 != null ? `${(d.var_95 * 100).toFixed(2)}%` : 'N/A'],
    ['CVaR 95', d.cvar_95 != null ? `${(d.cvar_95 * 100).toFixed(2)}%` : 'N/A'],
    ['Sharpe', d.sharpe != null ? d.sharpe.toFixed(2) : 'N/A'],
    ['Vol forecast', d.volatility_forecast != null ? `${(d.volatility_forecast * 100).toFixed(2)}%` : 'N/A'],
    ['Fundamental', d.fundamental_score != null ? d.fundamental_score.toFixed(2) : 'N/A'],
    ['Position size', d.position_size != null ? `${(d.position_size * 100).toFixed(1)}%` : 'N/A'],
    ['Actual return', d.actual_return != null ? `${(d.actual_return * 100).toFixed(2)}%` : 'pending'],
    ['Correct', d.correct == null ? 'pending' : d.correct === 1 ? 'yes' : 'no'],
  ]
  return (
    <>
      <tr className={`border-b border-border/50 hover:bg-muted/50 cursor-pointer ${expanded ? 'bg-muted/60' : ''}`} onClick={onToggle}>
        <td className="py-2 px-3 font-mono text-xs">
          {d.date} <span className="text-muted-foreground text-[10px]">{expanded ? '▲' : '▼'}</span>
        </td>
        <td className="py-2 px-3 font-mono text-xs">{d.ticker}</td>
        <td className="py-2 px-3">
          <Badge variant={d.action === 'BUY' ? 'success' : d.action === 'SELL' ? 'danger' : 'default'}>
            {d.action}
          </Badge>
        </td>
        <td className="py-2 px-3 font-mono text-xs text-right">
          {d.confidence ? `${(d.confidence * 100).toFixed(1)}%` : '-'}
        </td>
        <td className="py-2 px-3 text-xs">{d.regime || '-'}</td>
        <td className="py-2 px-3">
          <Badge variant={d.source === 'live' ? 'success' : d.source === 'backfill' ? 'warning' : 'default'}>
            {d.source || 'unknown'}
          </Badge>
        </td>
        <td className="py-2 px-3 text-xs text-muted-foreground max-w-[200px] truncate">{d.reasoning}</td>
      </tr>
      {expanded && (
        <tr className="bg-muted/30">
          <td colSpan={7} className="p-4">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
              {rows.map(([label, value]) => (
                <div key={label} className="rounded border border-border/60 px-2 py-1.5 text-xs">
                  <p className="font-mono uppercase text-[10px] text-muted-foreground">{label}</p>
                  <p className="font-mono font-semibold">{value}</p>
                </div>
              ))}
            </div>
            {d.reasoning && (
              <p className="mt-3 text-xs text-muted-foreground border-t border-border/50 pt-2">{d.reasoning}</p>
            )}
          </td>
        </tr>
      )}
    </>
  )
}

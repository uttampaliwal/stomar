import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState, PageHeader } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'
import type { PaperPositionSummary, PaperPositionsResponse, PaperTradesResponse, PaperOrderResponse, PaperCloseResponse, PaperResetResponse, PaperPerformanceResponse } from '../lib/api-types'

export default function PaperTrading() {
  const { data: state, loading, error, refetch } = useApi<PaperPositionSummary>('/api/paper-trading/state')
  const { data: positions, refetch: refetchPositions } = useApi<PaperPositionsResponse>('/api/paper-trading/positions')
  const { data: trades } = useApi<PaperTradesResponse>('/api/paper-trading/trades')
  const { data: stocks } = useApi<{ stocks: string[] }>('/api/paper-trading/stocks')
  const { data: perf } = useApi<PaperPerformanceResponse>('/api/paper-trading/performance')
  const { post: placeOrder, loading: ordering } = usePostApi<PaperOrderResponse>('/api/paper-trading/order')
  const { post: closePosition, loading: closing } = usePostApi<PaperCloseResponse>('/api/paper-trading/close-position')
  const { post: resetAccount, loading: resetting } = usePostApi<PaperResetResponse>('/api/paper-trading/reset')

  const [form, setForm] = useState({ ticker: 'RELIANCE.NS', side: 'BUY', quantity: 1, order_type: 'MARKET', limit_price: '' })
  const [resetCapital, setResetCapital] = useState(200000)

  const handleOrder = async () => {
    await placeOrder(form)
    refetch()
    refetchPositions()
  }

  const handleClose = async (ticker: string) => {
    await closePosition({ ticker })
    refetch()
    refetchPositions()
  }

  const handleReset = async () => {
    await resetAccount({ capital: resetCapital })
    refetch()
    refetchPositions()
  }

  return (
    <div className="space-y-6">
      <PageHeader
        title="Paper Trading"
        description="Simulated execution with real prices and built-in risk controls"
        badge="Live"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-sm font-semibold">CLI equivalent</p>
            <p className="text-sm text-muted-foreground">The same workflow can be launched from the terminal for repeatable paper trading.</p>
          </div>
          <div className="rounded-lg border border-border bg-background/70 px-3 py-2 font-mono text-xs text-muted-foreground">
            python run_daily.py --paper-trade --capital 200000
          </div>
        </div>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {state && !('error' in state) && (
        <>
          {/* Account Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Equity" value={formatCurrency(state.current_equity || 0)} trend="up" />
            <Stat label="Cash" value={formatCurrency(state.cash || 0)} />
            <Stat label="Realized P&L" value={formatCurrency(state.realized_pnl || 0)} trend={state.realized_pnl >= 0 ? 'up' : 'down'} />
            <Stat label="Unrealized P&L" value={formatCurrency(state.unrealized_pnl || 0)} trend={state.unrealized_pnl >= 0 ? 'up' : 'down'} />
          </div>

          <Card className={`space-y-3 border-l-4 ${state.risk_status?.halted ? 'border-l-rose bg-rose/5' : 'border-l-emerald bg-emerald/5'}`}>
            <div className="flex items-center justify-between">
              <p className="text-sm font-semibold">Risk Controls</p>
              <Badge variant={state.risk_status?.halted ? 'danger' : 'success'}>
                {state.risk_status?.halted ? 'TRADING HALTED' : 'NORMAL'}
              </Badge>
            </div>
            {state.risk_status?.halted && (
              <p className="text-sm text-rose font-medium">
                {state.risk_status.halt_reason || 'Risk limit breached'}
              </p>
            )}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs">
              <div>
                <p className="text-muted-foreground">Drawdown</p>
                <p className={`font-mono font-semibold ${((state.risk_status?.drawdown_pct || 0) * 100) > 10 ? 'text-rose' : 'text-foreground'}`}>
                  {((state.risk_status?.drawdown_pct || 0) * 100).toFixed(1)}%
                </p>
              </div>
              <div>
                <p className="text-muted-foreground">Daily Loss Remaining</p>
                <p className="font-mono font-semibold">{formatCurrency(state.risk_status?.daily_loss_remaining || 0)}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Peak Equity</p>
                <p className="font-mono font-semibold">{formatCurrency(state.risk_status?.peak_equity || 0)}</p>
              </div>
              <div>
                <p className="text-muted-foreground">Consecutive Losses</p>
                <p className={`font-mono font-semibold ${(state.risk_status?.consecutive_losses || 0) >= 3 ? 'text-amber' : 'text-foreground'}`}>
                  {state.risk_status?.consecutive_losses || 0} / 5
                </p>
              </div>
            </div>
            <div className="flex items-center gap-2 pt-2 border-t border-border/50">
              <label className="text-xs text-muted-foreground">Reset Capital (₹)</label>
              <input
                type="number"
                value={resetCapital}
                onChange={(e) => setResetCapital(parseInt(e.target.value) || 200000)}
                className="w-28 rounded border border-border bg-background px-2 py-1 text-xs font-mono"
              />
              <button
                onClick={handleReset}
                disabled={resetting}
                className="rounded bg-amber/10 text-amber border border-amber/20 px-2 py-1 text-xs hover:bg-amber/20 transition-colors disabled:opacity-50"
              >
                {resetting ? 'Resetting...' : 'Reset Account'}
              </button>
            </div>
          </Card>

          {/* Order Form */}
          <SectionHeader title="Place Order" />
          <Card>
            <div className="flex flex-col gap-3 md:flex-row md:items-end">
              <div className="flex-1">
                <label className="text-xs text-muted-foreground block mb-1">Ticker</label>
                <select
                  value={form.ticker}
                  onChange={(e) => setForm({ ...form, ticker: e.target.value })}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                >
                  {stocks?.stocks?.map((s: string) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div className="w-full md:w-32">
                <label className="text-xs text-muted-foreground block mb-1">Side</label>
                <select
                  value={form.side}
                  onChange={(e) => setForm({ ...form, side: e.target.value })}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div className="w-full md:w-28">
                <label className="text-xs text-muted-foreground block mb-1">Quantity</label>
                <input
                  type="number"
                  value={form.quantity}
                  onChange={(e) => setForm({ ...form, quantity: parseInt(e.target.value) || 0 })}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                />
              </div>
              <div className="w-full md:w-32">
                <label className="text-xs text-muted-foreground block mb-1">Order Type</label>
                <select
                  value={form.order_type}
                  onChange={(e) => setForm({ ...form, order_type: e.target.value })}
                  className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                >
                  <option value="MARKET">MARKET</option>
                  <option value="LIMIT">LIMIT</option>
                </select>
              </div>
              {form.order_type === 'LIMIT' && (
                <div className="w-full md:w-32">
                  <label className="text-xs text-muted-foreground block mb-1">Limit Price (₹)</label>
                  <input
                    type="number"
                    value={form.limit_price}
                    onChange={(e) => setForm({ ...form, limit_price: e.target.value })}
                    placeholder="0.00"
                    className="w-full rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                  />
                </div>
              )}
              <button
                onClick={handleOrder}
                disabled={ordering}
                className="rounded-lg bg-cyan px-4 py-2 text-sm font-semibold text-black hover:bg-cyan/80 transition-colors disabled:opacity-50"
              >
                {ordering ? 'Placing...' : 'Execute'}
              </button>
            </div>
          </Card>

          {/* Positions */}
          {positions && positions.positions.length > 0 ? (
            <>
              <SectionHeader title="Open Positions" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                      <th className="text-left py-2">Ticker</th>
                      <th className="text-right py-2">Qty</th>
                      <th className="text-right py-2">Avg Cost</th>
                      <th className="text-right py-2">LTP</th>
                      <th className="text-right py-2">P&L</th>
                      <th className="text-right py-2">P&L %</th>
                      <th className="text-center py-2">Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {positions.positions.map((p) => (
                      <tr key={p.ticker} className="border-b border-border/50 hover:bg-accent/30">
                        <td className="py-2.5 font-mono font-semibold">{p.ticker}</td>
                        <td className="py-2.5 text-right font-mono">{p.quantity}</td>
                        <td className="py-2.5 text-right font-mono">₹{p.avg_cost}</td>
                        <td className="py-2.5 text-right font-mono">₹{p.current_price}</td>
                        <td className={`py-2.5 text-right font-mono ${p.pnl >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {formatCurrency(p.pnl)}
                        </td>
                        <td className={`py-2.5 text-right font-mono ${p.pnl_pct >= 0 ? 'text-emerald' : 'text-rose'}`}>
                          {p.pnl_pct >= 0 ? '+' : ''}{p.pnl_pct}%
                        </td>
                        <td className="py-2.5 text-center">
                          <button
                            onClick={() => handleClose(p.ticker)}
                            disabled={closing}
                            className="rounded bg-rose/10 text-rose border border-rose/20 px-2 py-1 text-xs hover:bg-rose/20 transition-colors disabled:opacity-50"
                          >
                            Close
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          ) : (
            <EmptyState message="No open positions yet. Place an order to start the simulation." />
          )}

          {/* Trade Log */}
          {trades && trades.trades.length > 0 ? (
            <>
              <SectionHeader title="Recent Trades" />
              <Card className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-border text-xs text-muted-foreground uppercase">
                      <th className="text-left py-2">Date</th>
                      <th className="text-left py-2">Ticker</th>
                      <th className="text-center py-2">Side</th>
                      <th className="text-right py-2">Qty</th>
                      <th className="text-right py-2">Price</th>
                    </tr>
                  </thead>
                  <tbody>
                    {trades.trades.slice().reverse().slice(0, 20).map((t, i) => (
                      <tr key={i} className="border-b border-border/50 hover:bg-accent/30">
                        <td className="py-2.5 font-mono text-xs text-muted-foreground">{t.date}</td>
                        <td className="py-2.5 font-mono font-semibold">{t.ticker}</td>
                        <td className="py-2.5 text-center">
                          <Badge variant={t.side === 'BUY' ? 'success' : 'danger'}>{t.side}</Badge>
                        </td>
                        <td className="py-2.5 text-right font-mono">{t.quantity}</td>
                        <td className="py-2.5 text-right font-mono">₹{t.price}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          ) : (
            <EmptyState message="The trade log will appear here after your first fill." />
          )}

          {/* P3.4: Performance — rolling accuracy, signal breakdown, drawdown */}
          {perf && !('error' in perf) && (
            <>
              <SectionHeader title="Performance vs Signal Honesty (P3.4)" />
              <Card>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                  <Stat
                    label={`Rolling ${perf.rolling_accuracy_30d?.days || 30}-day Accuracy`}
                    value={perf.rolling_accuracy_30d?.accuracy != null
                      ? `${(perf.rolling_accuracy_30d.accuracy * 100).toFixed(1)}%`
                      : 'N/A'}
                    trend={(perf.rolling_accuracy_30d?.accuracy || 0) >= 0.51 ? 'up' : 'down'}
                  />
                  <Stat
                    label="Current Drawdown"
                    value={perf.drawdown?.current_drawdown_pct != null
                      ? `${(perf.drawdown.current_drawdown_pct * 100).toFixed(1)}%`
                      : 'N/A'}
                    trend={((perf.drawdown?.current_drawdown_pct || 0) > 0.10) ? 'down' : 'up'}
                  />
                  <Stat
                    label="Max Drawdown (all time)"
                    value={perf.drawdown?.max_drawdown_pct != null
                      ? `${(perf.drawdown.max_drawdown_pct * 100).toFixed(1)}%`
                      : 'N/A'}
                    trend={((perf.drawdown?.max_drawdown_pct || 0) > 0.10) ? 'down' : 'up'}
                  />
                  <Stat
                    label="Resolved Live Decisions (30d)"
                    value={perf.rolling_accuracy_30d?.n_resolved ?? 0}
                  />
                </div>
                <p className="mt-2 text-xs text-muted-foreground">
                  Rolling accuracy counts only resolved live BUY/SELL/HOLD decisions with actual outcomes —
                  backfill is excluded because it is in-sample by construction.
                </p>
                {((perf.drawdown?.current_drawdown_pct || 0) > 0.10) && (
                  <p className="mt-2 text-xs font-semibold text-rose">
                    ALERT: current drawdown exceeds 10% — review strategy before continuing.
                  </p>
                )}
              </Card>

              {perf.drawdown?.equity_curve?.length > 2 && (
                <Card>
                  <p className="mb-2 text-xs text-muted-foreground">Equity curve (last {perf.drawdown.equity_curve.length} snapshots)</p>
                  <div className="h-40 flex items-end gap-px">
                    {perf.drawdown.equity_curve.map((p) => {
                      const min = Math.min(...perf.drawdown!.equity_curve!.map((x) => x.equity))
                      const max = Math.max(...perf.drawdown!.equity_curve!.map((x) => x.equity))
                      const h = max > min ? ((p.equity - min) / (max - min)) * 100 : 50
                      return (
                        <div
                          key={p.date}
                          title={`${p.date}: ${formatCurrency(p.equity)}`}
                          className={`flex-1 rounded-t ${p.equity >= perf.drawdown!.peak_equity! ? 'bg-emerald/60' : 'bg-cyan/60'}`}
                          style={{ height: `${Math.max(h, 2)}%` }}
                        />
                      )
                    })}
                  </div>
                </Card>
              )}

              <Card>
                <p className="mb-2 text-sm font-semibold">Signal Accuracy by Module (live decisions)</p>
                {perf.signal_accuracy && Object.keys(perf.signal_accuracy).length > 0 ? (
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3 text-sm">
                    {Object.entries(perf.signal_accuracy).map(([name, acc]) => (
                      <div key={name} className="flex items-center justify-between rounded border border-border/60 px-3 py-2">
                        <span className="text-xs font-mono uppercase text-muted-foreground">{name}</span>
                        <span className={`font-mono font-semibold ${acc == null ? '' : acc >= 0.51 ? 'text-emerald' : 'text-rose'}`}>
                          {acc == null ? 'N/A' : `${(acc * 100).toFixed(0)}%`}
                        </span>
                      </div>
                    ))}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">No resolved live decisions yet — appears after live paper runs accumulate.</p>
                )}
              </Card>
            </>
          )}
        </>
      )}
    </div>
  )
}

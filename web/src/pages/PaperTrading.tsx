import { useState } from 'react'
import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, Badge, EmptyState } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'

export default function PaperTrading() {
  const { data: state, loading, error, refetch } = useApi<any>('/api/paper-trading/state')
  const { data: positions } = useApi<any>('/api/paper-trading/positions')
  const { data: trades } = useApi<any>('/api/paper-trading/trades')
  const { data: stocks } = useApi<{ stocks: string[] }>('/api/paper-trading/stocks')
  const { post: placeOrder, loading: ordering } = usePostApi<any>('/api/paper-trading/order')

  const [form, setForm] = useState({ ticker: 'RELIANCE.NS', side: 'BUY', quantity: 1 })

  const handleOrder = async () => {
    await placeOrder(form)
    refetch()
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Paper Trading</h1>
        <p className="text-sm text-muted-foreground">Simulated execution with real prices</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {state && !state.error && (
        <>
          {/* Account Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            <Stat label="Equity" value={formatCurrency(state.current_equity || state.equity || 0)} trend="up" />
            <Stat label="Cash" value={formatCurrency(state.cash || 0)} />
            <Stat label="Realized P&L" value={formatCurrency(state.realized_pnl || 0)} trend={state.realized_pnl >= 0 ? 'up' : 'down'} />
            <Stat label="Unrealized P&L" value={formatCurrency(state.unrealized_pnl || 0)} trend={state.unrealized_pnl >= 0 ? 'up' : 'down'} />
          </div>

          {/* Order Form */}
          <SectionHeader title="Place Order" />
          <Card>
            <div className="flex gap-3 items-end">
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Ticker</label>
                <select
                  value={form.ticker}
                  onChange={(e) => setForm({ ...form, ticker: e.target.value })}
                  className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                >
                  {stocks?.stocks?.map((s: string) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Side</label>
                <select
                  value={form.side}
                  onChange={(e) => setForm({ ...form, side: e.target.value })}
                  className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono"
                >
                  <option value="BUY">BUY</option>
                  <option value="SELL">SELL</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-muted-foreground block mb-1">Quantity</label>
                <input
                  type="number"
                  value={form.quantity}
                  onChange={(e) => setForm({ ...form, quantity: parseInt(e.target.value) || 0 })}
                  className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono w-24"
                />
              </div>
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
          {positions?.positions?.length > 0 && (
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
                    </tr>
                  </thead>
                  <tbody>
                    {positions.positions.map((p: any) => (
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
                      </tr>
                    ))}
                  </tbody>
                </table>
              </Card>
            </>
          )}

          {/* Trade Log */}
          {trades?.trades?.length > 0 && (
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
                    {trades.trades.slice().reverse().slice(0, 20).map((t: any, i: number) => (
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
          )}
        </>
      )}
    </div>
  )
}

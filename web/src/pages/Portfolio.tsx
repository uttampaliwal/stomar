import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, PageHeader, EmptyState } from '@/components/UI'
import { useApi, usePostApi } from '@/hooks/useApi'
import { formatCurrency } from '@/lib/utils'

export default function Portfolio() {
  const stats = useApi<any>('/api/portfolio/stats')
  const trades = useApi<any>('/api/portfolio/trades')
  const { post: closePosition, loading: closing } = usePostApi<any>('/api/paper-trading/close-position')

  const handleExit = async (ticker: string) => {
    await closePosition({ ticker })
    stats.refetch()
    trades.refetch()
  }

  if (stats.loading || trades.loading) return <Spinner />
  if (stats.error) return <ErrorDisplay message={stats.error} />

  const data = stats.data || {}
  const tradeData = trades.data?.trades || []

  return (
    <div className="space-y-6">
      <PageHeader
        title="Portfolio"
        description="Paper trading portfolio overview"
        badge="Live"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          This page mirrors the paper-trading account state and stays aligned with the daily automation workflow.
        </p>
      </Card>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
        <Stat
          label="Total Equity"
          value={formatCurrency(data.equity || 0)}
          sub={`Cash: ${formatCurrency(data.cash || 0)}`}
          trend="up"
        />
        <Stat
          label="Unrealized P&L"
          value={formatCurrency(data.unrealized_pnl || 0)}
          sub={data.positions?.length ? `${data.positions.length} positions` : 'No positions'}
          trend={data.unrealized_pnl >= 0 ? 'up' : 'down'}
        />
        <Stat
          label="Realized P&L"
          value={formatCurrency(data.realized_pnl || 0)}
          sub={`${data.total_trades || 0} total trades`}
          trend={data.realized_pnl >= 0 ? 'up' : 'down'}
        />
        <Stat
          label="Sharpe Ratio"
          value={data.sharpe_ratio || '0.00'}
          sub="Risk-Adjusted"
          trend={data.sharpe_ratio > 1 ? 'up' : data.sharpe_ratio < 0 ? 'down' : 'neutral'}
        />
        <Stat
          label="Win Rate"
          value={`${data.win_rate || 0}%`}
          sub={`${data.closed_positions || 0} closed`}
          trend="neutral"
        />
      </div>

      {/* Positions Table */}
      <Card>
        <SectionHeader title="Open Positions" />
        {data.positions && data.positions.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 font-medium text-muted-foreground">Ticker</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">Qty</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">Avg Cost</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">Current</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">P&L</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">P&L %</th>
                  <th className="text-center py-2 px-3 font-medium text-muted-foreground">Action</th>
                </tr>
              </thead>
              <tbody>
                {data.positions.map((pos: any, i: number) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                    <td className="py-2 px-3 font-medium">{pos.ticker}</td>
                    <td className="py-2 px-3 text-right">{pos.quantity}</td>
                    <td className="py-2 px-3 text-right">{formatCurrency(pos.avg_cost)}</td>
                    <td className="py-2 px-3 text-right">{formatCurrency(pos.current_price)}</td>
                    <td className={`py-2 px-3 text-right font-medium ${pos.pnl >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {formatCurrency(pos.pnl)}
                    </td>
                    <td className={`py-2 px-3 text-right font-medium ${(pos.pnl_pct || 0) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {(pos.pnl_pct || 0) >= 0 ? '+' : ''}{pos.pnl_pct || 0}%
                    </td>
                    <td className="py-2 px-3 text-center">
                      <button
                        onClick={() => handleExit(pos.ticker)}
                        disabled={closing}
                        className="rounded bg-rose/10 text-rose border border-rose/20 px-2 py-1 text-xs hover:bg-rose/20 transition-colors disabled:opacity-50"
                      >
                        Exit
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState message="No open positions yet. Place an order to start the simulation." />
        )}
      </Card>

      {/* Recent Trades */}
      <Card>
        <SectionHeader title="Recent Trades" />
        {tradeData.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-border">
                  <th className="text-left py-2 px-3 font-medium text-muted-foreground">Date</th>
                  <th className="text-left py-2 px-3 font-medium text-muted-foreground">Ticker</th>
                  <th className="text-left py-2 px-3 font-medium text-muted-foreground">Side</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">Qty</th>
                  <th className="text-right py-2 px-3 font-medium text-muted-foreground">Price</th>
                </tr>
              </thead>
              <tbody>
                {tradeData.map((trade: any, i: number) => (
                  <tr key={i} className="border-b border-border/50 hover:bg-muted/50">
                    <td className="py-2 px-3 text-muted-foreground">{trade.date?.split('T')[0] || '—'}</td>
                    <td className="py-2 px-3 font-medium">{trade.ticker}</td>
                    <td className={`py-2 px-3 font-medium ${trade.side === 'BUY' ? 'text-emerald' : 'text-rose'}`}>
                      {trade.side}
                    </td>
                    <td className="py-2 px-3 text-right">{trade.quantity}</td>
                    <td className="py-2 px-3 text-right">{formatCurrency(trade.price)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <EmptyState message="No trades recorded yet." />
        )}
      </Card>
    </div>
  )
}

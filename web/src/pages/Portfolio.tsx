import { Card, Stat, SectionHeader, Spinner, ErrorDisplay } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { formatCurrency, formatPercent } from '@/lib/utils'
import { Briefcase, TrendingUp, TrendingDown, DollarSign, Activity } from 'lucide-react'

export default function Portfolio() {
  const stats = useApi<any>('/api/portfolio/stats')
  const trades = useApi<any>('/api/portfolio/trades')

  if (stats.loading || trades.loading) return <Spinner />
  if (stats.error) return <ErrorDisplay message={stats.error} />

  const data = stats.data || {}
  const tradeData = trades.data?.trades || []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Portfolio</h1>
          <p className="text-sm text-muted-foreground mt-1">Paper trading portfolio overview</p>
        </div>
        <div className="flex items-center gap-2">
          <Briefcase className="h-5 w-5 text-muted-foreground" />
        </div>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
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
          label="Open Positions"
          value={data.positions?.length || 0}
          sub="Active trades"
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
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="py-8 text-center text-muted-foreground">
            No open positions
          </div>
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
          <div className="py-8 text-center text-muted-foreground">
            No trades recorded
          </div>
        )}
      </Card>
    </div>
  )
}

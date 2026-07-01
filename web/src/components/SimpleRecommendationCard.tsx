import { ArrowUpRight, ArrowDownRight, MinusCircle } from 'lucide-react'

interface RecommendationCardProps {
  data?: {
    action?: string
    confidence?: number
    rationale?: string
    risk_level?: string
    price?: number
    trend_pct?: number
    volatility?: number
  }
  loading?: boolean
}

export function SimpleRecommendationCard({ data, loading }: RecommendationCardProps) {
  if (loading) {
    return <div className="rounded-2xl border border-border bg-card p-4 text-sm text-muted-foreground">Loading recommendation…</div>
  }

  const action = data?.action || 'HOLD'
  const icon = action === 'BUY' ? <ArrowUpRight className="h-5 w-5" /> : action === 'SELL' ? <ArrowDownRight className="h-5 w-5" /> : <MinusCircle className="h-5 w-5" />

  return (
    <div className="rounded-2xl border border-border bg-card p-5 shadow-sm">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-sm text-muted-foreground">What should I do today?</p>
          <div className="flex items-center gap-2">
            {icon}
            <h3 className="text-xl font-semibold">{action}</h3>
          </div>
        </div>
        <div className="rounded-full border border-border px-3 py-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">
          {data?.risk_level || 'LOW'} risk
        </div>
      </div>
      <p className="mt-4 text-sm text-muted-foreground">{data?.rationale || 'The system is waiting for a clearer signal.'}</p>
      <div className="mt-4 grid grid-cols-3 gap-3 text-sm">
        <div className="rounded-xl bg-background/70 p-3">
          <div className="text-xs text-muted-foreground">Confidence</div>
          <div className="font-semibold">{((data?.confidence || 0) * 100).toFixed(0)}%</div>
        </div>
        <div className="rounded-xl bg-background/70 p-3">
          <div className="text-xs text-muted-foreground">Price</div>
          <div className="font-semibold">₹{data?.price?.toFixed(2) || '—'}</div>
        </div>
        <div className="rounded-xl bg-background/70 p-3">
          <div className="text-xs text-muted-foreground">Trend</div>
          <div className="font-semibold">{data?.trend_pct?.toFixed(2) || '0.00'}%</div>
        </div>
      </div>
      <p className="mt-4 text-xs text-muted-foreground">This uses historical market data and a conservative signal model. It is educational, not a guarantee of profit.</p>
    </div>
  )
}

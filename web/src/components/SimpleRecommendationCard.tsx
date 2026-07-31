import { ArrowUpRight, ArrowDownRight, MinusCircle } from 'lucide-react'
import { cn } from '@/lib/utils'

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
    return <div className="rounded-lg border border-border bg-card/60 p-4 text-sm text-muted-foreground font-mono">Loading recommendation...</div>
  }

  const action = data?.action || 'HOLD'
  const actionColor = action === 'BUY' ? 'text-emerald' : action === 'SELL' ? 'text-rose' : 'text-amber'
  const actionBg = action === 'BUY' ? 'bg-emerald/10 border-emerald/30' : action === 'SELL' ? 'bg-rose/10 border-rose/30' : 'bg-amber/10 border-amber/30'
  const icon = action === 'BUY' ? <ArrowUpRight className="h-4 w-4" /> : action === 'SELL' ? <ArrowDownRight className="h-4 w-4" /> : <MinusCircle className="h-4 w-4" />

  return (
    <div className="rounded-lg border border-border bg-card/60 p-4">
      <div className="flex items-center justify-between gap-3">
        <div>
          <p className="text-[0.6rem] font-mono uppercase tracking-widest text-muted-foreground">Decision</p>
          <div className={cn('flex items-center gap-2 mt-1', actionColor)}>
            {icon}
            <h3 className="text-lg font-bold font-mono">{action}</h3>
          </div>
        </div>
        <div className={cn('rounded-md border px-2.5 py-1 text-[0.6rem] font-mono font-bold uppercase', actionBg, actionColor)}>
          {data?.risk_level || 'LOW'} risk
        </div>
      </div>
      <p className="mt-3 text-xs text-muted-foreground leading-relaxed">{data?.rationale || 'The system is waiting for a clearer signal.'}</p>
      <div className="mt-3 grid grid-cols-3 gap-2 text-xs">
        <div className="rounded border border-border bg-background/50 p-2">
          <p className="text-[0.55rem] font-mono text-muted-foreground uppercase">Confidence</p>
          <p className="font-mono font-bold">{((data?.confidence || 0) * 100).toFixed(0)}%</p>
        </div>
        <div className="rounded border border-border bg-background/50 p-2">
          <p className="text-[0.55rem] font-mono text-muted-foreground uppercase">Price</p>
          <p className="font-mono font-bold">₹{data?.price?.toFixed(2) || '—'}</p>
        </div>
        <div className="rounded border border-border bg-background/50 p-2">
          <p className="text-[0.55rem] font-mono text-muted-foreground uppercase">Trend</p>
          <p className={cn('font-mono font-bold', (data?.trend_pct || 0) >= 0 ? 'text-emerald' : 'text-rose')}>
            {data?.trend_pct?.toFixed(2) || '0.00'}%
          </p>
        </div>
      </div>
      <p className="mt-2 text-[0.55rem] text-muted-foreground/60">Educational signal — not financial advice.</p>
    </div>
  )
}

import { cn } from '@/lib/utils'

// ─── Status Pill ───
interface StatusPillProps {
  label: string
  value: string
  status: 'online' | 'offline' | 'warning' | 'active'
  className?: string
}

const statusColors = {
  online: 'border-emerald/40 bg-emerald/10 text-emerald',
  offline: 'border-rose/40 bg-rose/10 text-rose',
  warning: 'border-amber/40 bg-amber/10 text-amber',
  active: 'border-cyan/40 bg-cyan/10 text-cyan',
}

const dotColors = {
  online: 'bg-emerald',
  offline: 'bg-rose',
  warning: 'bg-amber',
  active: 'bg-cyan',
}

export function StatusPill({ label, value, status, className }: StatusPillProps) {
  return (
    <div
      className={cn(
        'flex items-center gap-2 rounded-md border px-2.5 py-1 font-mono text-[0.65rem]',
        statusColors[status],
        className,
      )}
    >
      <span className={cn('h-1.5 w-1.5 rounded-full', dotColors[status], status === 'active' && 'animate-pulse')} />
      <span className="uppercase tracking-wider opacity-70">{label}:</span>
      <span className="font-bold">{value}</span>
    </div>
  )
}

// ─── Risk Gauge ───
interface RiskGaugeProps {
  value: number // 0-100
  label?: string
  size?: 'sm' | 'md' | 'lg'
  className?: string
}

export function RiskGauge({ value, label = 'RISK', size = 'md', className }: RiskGaugeProps) {
  const clamped = Math.max(0, Math.min(100, value))
  const color =
    clamped < 30 ? 'text-emerald' : clamped < 60 ? 'text-amber' : 'text-rose'
  const barColor =
    clamped < 30 ? 'bg-emerald' : clamped < 60 ? 'bg-amber' : 'bg-rose'
  const sizeMap = { sm: 'h-1', md: 'h-1.5', lg: 'h-2' }

  return (
    <div className={cn('space-y-1', className)}>
      <div className="flex items-center justify-between">
        <span className="font-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">{label}</span>
        <span className={cn('font-mono text-xs font-bold', color)}>{clamped.toFixed(0)}%</span>
      </div>
      <div className={cn('w-full rounded-full bg-muted/30', sizeMap[size])}>
        <div
          className={cn('rounded-full transition-all duration-500', barColor, sizeMap[size])}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  )
}

// ─── Consensus Card ───
interface ConsensusCardProps {
  ticker: string
  signal: string
  confidence: number
  regime?: string
  className?: string
}

export function ConsensusCard({ ticker, signal, confidence, regime, className }: ConsensusCardProps) {
  const signalColor =
    signal === 'BUY' ? 'text-emerald border-emerald/30 bg-emerald/10' :
    signal === 'SELL' ? 'text-rose border-rose/30 bg-rose/10' :
    'text-amber border-amber/30 bg-amber/10'

  const arrow = signal === 'BUY' ? '▲' : signal === 'SELL' ? '▼' : '—'

  return (
    <div className={cn('flex items-center justify-between rounded-lg border border-border bg-card/60 px-3 py-2 transition-all hover:border-cyan/20', className)}>
      <div className="flex items-center gap-3">
        <span className="font-mono text-xs font-bold text-foreground">{ticker.replace('.NS', '')}</span>
        {regime && (
          <span className="rounded px-1.5 py-0.5 text-[0.55rem] font-mono uppercase bg-muted/50 text-muted-foreground">
            {regime}
          </span>
        )}
      </div>
      <div className="flex items-center gap-2">
        <span className={cn('rounded-md border px-2 py-0.5 text-[0.6rem] font-mono font-bold', signalColor)}>
          {arrow} {signal}
        </span>
        <span className="font-mono text-[0.65rem] text-muted-foreground w-10 text-right">
          {(confidence * 100).toFixed(0)}%
        </span>
      </div>
    </div>
  )
}

// ─── Ticker Marquee ───
interface TickerItem {
  symbol: string
  price: number
  change: number
  changePct: number
}

interface TickerMarqueeProps {
  items: TickerItem[]
  className?: string
}

export function TickerMarquee({ items, className }: TickerMarqueeProps) {
  return (
    <div className={cn('flex items-center gap-4 overflow-hidden border-b border-border bg-card/40 py-1.5 font-mono text-[0.65rem]', className)}>
      <div className="flex animate-[marquee_30s_linear_infinite] gap-4 whitespace-nowrap">
        {[...items, ...items].map((item, i) => (
          <span key={i} className="flex items-center gap-1.5">
            <span className="font-bold text-foreground">{item.symbol}</span>
            <span className="text-muted-foreground">₹{item.price.toFixed(1)}</span>
            <span className={item.changePct >= 0 ? 'text-emerald' : 'text-rose'}>
              {item.changePct >= 0 ? '+' : ''}{item.changePct.toFixed(2)}%
            </span>
          </span>
        ))}
      </div>
    </div>
  )
}

// ─── ML Consensus Card ───
interface ModelVote {
  name: string
  direction: 'UP' | 'DN'
  confidence: number
}

interface MLConsensusCardProps {
  modelVotes: ModelVote[]
  ensembleDirection: string
  ensembleConfidence: number
  className?: string
}

export function MLConsensusCard({ modelVotes, ensembleDirection, ensembleConfidence, className }: MLConsensusCardProps) {
  return (
    <div className={cn('rounded-lg border border-border bg-card/60 p-3', className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">ML Ensemble</span>
        <span className={cn(
          'rounded-md border px-2 py-0.5 text-[0.6rem] font-mono font-bold',
          ensembleDirection === 'BUY'
            ? 'text-emerald border-emerald/30 bg-emerald/10'
            : 'text-rose border-rose/30 bg-rose/10',
        )}>
          {ensembleDirection === 'BUY' ? '▲' : '▼'} {ensembleDirection}
        </span>
      </div>
      <div className="space-y-1.5">
        {modelVotes.map((vote) => (
          <div key={vote.name} className="flex items-center gap-2">
            <span className="w-20 text-[0.6rem] font-mono text-muted-foreground">{vote.name}</span>
            <div className="flex-1 h-1 rounded-full bg-muted/30 overflow-hidden">
              <div
                className={cn('h-full rounded-full', vote.direction === 'UP' ? 'bg-emerald' : 'bg-rose')}
                style={{ width: `${vote.confidence * 100}%` }}
              />
            </div>
            <span className={cn('w-8 text-right text-[0.6rem] font-mono font-bold', vote.direction === 'UP' ? 'text-emerald' : 'text-rose')}>
              {vote.direction === 'UP' ? '▲' : '▼'}
            </span>
          </div>
        ))}
      </div>
      <div className="mt-2 flex items-center justify-between border-t border-border pt-2">
        <span className="text-[0.6rem] font-mono text-muted-foreground">Ensemble Confidence</span>
        <span className="font-mono text-xs font-bold text-cyan">{(ensembleConfidence * 100).toFixed(1)}%</span>
      </div>
    </div>
  )
}

// ─── Order Pad Drawer ───
interface OrderPadProps {
  ticker: string
  currentPrice: number
  onClose?: () => void
  className?: string
}

export function OrderPad({ ticker, currentPrice, onClose, className }: OrderPadProps) {
  const estimatedCosts = {
    brokerage: currentPrice * 0.0003,
    stt: currentPrice * 0.001,
    exchangeCharge: currentPrice * 0.0000345,
    sebiFees: currentPrice * 0.000001,
    stampDuty: currentPrice * 0.00015,
    gst: (currentPrice * 0.0003 + currentPrice * 0.0000345) * 0.18,
  }
  const totalCost = Object.values(estimatedCosts).reduce((a, b) => a + b, 0)

  return (
    <div className={cn('rounded-lg border border-border bg-card/80 p-3', className)}>
      <div className="flex items-center justify-between mb-3">
        <span className="font-mono text-xs font-bold text-foreground">ORDER PAD</span>
        {onClose && (
          <button onClick={onClose} className="text-muted-foreground hover:text-foreground text-xs">✕</button>
        )}
      </div>
      <div className="space-y-2 text-[0.65rem] font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Ticker</span>
          <span className="font-bold">{ticker}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Market Price</span>
          <span>₹{currentPrice.toFixed(2)}</span>
        </div>
        <div className="border-t border-border pt-2 mt-2">
          <span className="text-muted-foreground uppercase tracking-wider text-[0.55rem]">NSE Cost Estimate (1 share)</span>
        </div>
        <div className="flex justify-between"><span className="text-muted-foreground">Brokerage</span><span>₹{estimatedCosts.brokerage.toFixed(4)}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">STT</span><span>₹{estimatedCosts.stt.toFixed(4)}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Exchange</span><span>₹{estimatedCosts.exchangeCharge.toFixed(4)}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">SEBI</span><span>₹{estimatedCosts.sebiFees.toFixed(4)}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">Stamp</span><span>₹{estimatedCosts.stampDuty.toFixed(4)}</span></div>
        <div className="flex justify-between"><span className="text-muted-foreground">GST</span><span>₹{estimatedCosts.gst.toFixed(4)}</span></div>
        <div className="flex justify-between border-t border-border pt-2 mt-1">
          <span className="text-muted-foreground font-bold">Total Cost</span>
          <span className="text-amber font-bold">₹{totalCost.toFixed(4)}</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Effective Rate</span>
          <span className="text-amber">{((totalCost / currentPrice) * 100).toFixed(3)}%</span>
        </div>
      </div>
    </div>
  )
}

// ─── Portfolio Health Card ───
interface PortfolioHealthProps {
  equity: number
  cash: number
  pnl: number
  riskStatus: string
  className?: string
}

export function PortfolioHealthCard({ equity, cash, pnl, riskStatus, className }: PortfolioHealthProps) {
  const pnlColor = pnl >= 0 ? 'text-emerald' : 'text-rose'

  return (
    <div className={cn('rounded-lg border border-border bg-card/60 p-3', className)}>
      <div className="flex items-center justify-between mb-2">
        <span className="font-mono text-[0.6rem] uppercase tracking-widest text-muted-foreground">Portfolio Health</span>
        <span className={cn(
          'rounded px-1.5 py-0.5 text-[0.55rem] font-mono',
          riskStatus === 'Active' ? 'bg-emerald/10 text-emerald' : 'bg-rose/10 text-rose',
        )}>
          {riskStatus}
        </span>
      </div>
      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-[0.55rem] text-muted-foreground uppercase">Equity</p>
          <p className="font-mono text-xs font-bold">₹{(equity / 1000).toFixed(1)}K</p>
        </div>
        <div>
          <p className="text-[0.55rem] text-muted-foreground uppercase">Cash</p>
          <p className="font-mono text-xs font-bold">₹{(cash / 1000).toFixed(1)}K</p>
        </div>
        <div>
          <p className="text-[0.55rem] text-muted-foreground uppercase">P&L</p>
          <p className={cn('font-mono text-xs font-bold', pnlColor)}>
            {pnl >= 0 ? '+' : ''}₹{(pnl / 1000).toFixed(1)}K
          </p>
        </div>
      </div>
    </div>
  )
}

// ─── Sparkline ───
interface SparklineProps {
  data: number[]
  color?: string
  height?: number
  className?: string
}

export function Sparkline({ data, color = '#22d3ee', height = 30, className }: SparklineProps) {
  if (data.length < 2) return null

  const min = Math.min(...data)
  const max = Math.max(...data)
  const range = max - min || 1

  const points = data.map((v, i) => {
    const x = (i / (data.length - 1)) * 100
    const y = height - ((v - min) / range) * (height - 4) - 2
    return `${x},${y}`
  }).join(' ')

  return (
    <svg width="100%" height={height} viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" className={className}>
      <polyline
        points={points}
        fill="none"
        stroke={color}
        strokeWidth="1.5"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  )
}

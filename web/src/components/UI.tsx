import React from 'react'
import { cn } from '@/lib/utils'

// ─── Card ───
export function Card({ className, children, ...props }: React.HTMLAttributes<HTMLDivElement>) {
  return (
    <div className={cn('glass rounded-xl p-4 animate-fade-in', className)} {...props}>
      {children}
    </div>
  )
}

// ─── Stat Card ───
interface StatProps {
  label: string
  value: string | number
  sub?: string
  trend?: 'up' | 'down' | 'neutral'
  className?: string
}

export function Stat({ label, value, sub, trend, className }: StatProps) {
  const color = trend === 'up' ? 'text-emerald' : trend === 'down' ? 'text-rose' : 'text-foreground'
  return (
    <div className={cn('rounded-xl border border-border bg-card/50 p-3 transition-all hover:border-cyan/30', className)}>
      <p className="text-[0.65rem] font-semibold uppercase tracking-widest text-muted-foreground">{label}</p>
      <p className={cn('font-mono text-xl font-bold mt-1', color)}>{value}</p>
      {sub && <p className="text-xs text-muted-foreground mt-0.5">{sub}</p>}
    </div>
  )
}

// ─── Badge ───
interface BadgeProps {
  children: React.ReactNode
  variant?: 'default' | 'success' | 'danger' | 'warning' | 'info'
  className?: string
}

const badgeVariants = {
  default: 'bg-secondary text-secondary-foreground',
  success: 'bg-emerald/15 text-emerald border border-emerald/30',
  danger: 'bg-rose/15 text-rose border border-rose/30',
  warning: 'bg-amber/15 text-amber border border-amber/30',
  info: 'bg-cyan/15 text-cyan border border-cyan/30',
}

export function Badge({ children, variant = 'default', className }: BadgeProps) {
  return (
    <span className={cn('inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold uppercase tracking-wide', badgeVariants[variant], className)}>
      {children}
    </span>
  )
}

// ─── Skeleton ───
export function Skeleton({ className }: { className?: string }) {
  return <div className={cn('animate-pulse rounded-lg bg-muted', className)} />
}

// ─── Section Header ───
export function SectionHeader({ title, action }: { title: string; action?: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between mb-4">
      <div className="flex items-center gap-2">
        <div className="h-5 w-1 rounded-full bg-gradient-to-b from-cyan to-violet" />
        <h2 className="text-sm font-bold uppercase tracking-wider text-muted-foreground">{title}</h2>
      </div>
      {action}
    </div>
  )
}

// ─── Page Header ───
interface PageHeaderProps {
  title: string
  description?: string
  badge?: React.ReactNode
  children?: React.ReactNode
  className?: string
}

export function PageHeader({ title, description, badge, children, className }: PageHeaderProps) {
  return (
    <div className={cn('flex flex-col gap-3 md:flex-row md:items-end md:justify-between', className)}>
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <h1 className="text-2xl font-bold tracking-tight">{title}</h1>
          {badge && (
            <span className="rounded-full border border-cyan/20 bg-cyan/10 px-2.5 py-0.5 text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-cyan">
              {badge}
            </span>
          )}
        </div>
        {description && <p className="text-sm text-muted-foreground">{description}</p>}
      </div>
      {children}
    </div>
  )
}

// ─── Empty State ───
export function EmptyState({ message = 'No data available' }: { message?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-muted-foreground">
      <p className="font-mono text-sm">{message}</p>
    </div>
  )
}

// ─── Loading Spinner ───
export function Spinner({ size = 'md' }: { size?: 'sm' | 'md' | 'lg' }) {
  const sizeClass = size === 'sm' ? 'h-4 w-4' : size === 'lg' ? 'h-8 w-8' : 'h-6 w-6'
  return (
    <div className="flex items-center justify-center py-8">
      <div className={`${sizeClass} animate-spin rounded-full border-2 border-muted border-t-cyan`} />
    </div>
  )
}

// ─── Error Display ───
export function ErrorDisplay({ message }: { message: string }) {
  return (
    <div className="rounded-xl border border-rose/30 bg-rose/10 p-4 text-center">
      <p className="font-mono text-sm text-rose">{message}</p>
    </div>
  )
}

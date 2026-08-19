import React from 'react'
import { NavLink } from 'react-router'
import {
  Activity, BarChart3, Brain, Briefcase,
  ChevronLeft, ChevronRight, Coins, FileText,
  FlaskConical, Gauge, Heart, Layers,
  LogOut, Network, PieChart, RefreshCcw,
  ScanSearch, Shield, SlidersHorizontal, Sparkles, Target,
  Thermometer, Zap, BookOpen
} from 'lucide-react'
import { useTheme } from './ThemeProvider'
import { useAuth } from '@/lib/auth'

const navSections = [
  {
    title: 'CORE',
    items: [
      { path: '/', label: 'Dashboard', icon: LayoutDashboard },
      { path: '/wealth-goals', label: 'Wealth Goals', icon: Coins },
      { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
      { path: '/paper-trading', label: 'Paper Trading', icon: FileText },
      { path: '/ledger', label: 'Ledger', icon: BookOpen },
    ],
  },
  {
    title: 'INTELLIGENCE',
    items: [
      { path: '/predictions', label: 'Predictions', icon: Brain },
      { path: '/scanner', label: 'Scanner', icon: ScanSearch },
      { path: '/consensus', label: 'Consensus', icon: Target },
      { path: '/sentiment', label: 'Sentiment', icon: Sparkles },
      { path: '/market-pulse', label: 'Market Pulse', icon: Activity },
      { path: '/ranking', label: 'Ranking', icon: BarChart3 },
    ],
  },
  {
    title: 'RISK & ANALYTICS',
    items: [
      { path: '/risk', label: 'Risk', icon: Shield },
      { path: '/optimizer', label: 'Optimizer', icon: SlidersHorizontal },
      { path: '/volatility', label: 'Volatility', icon: Gauge },
      { path: '/regime', label: 'Regime', icon: Thermometer },
      { path: '/correlation', label: 'Correlation', icon: Network },
      { path: '/scenarios', label: 'Scenarios', icon: Layers },
    ],
  },
  {
    title: 'OPERATIONS',
    items: [
      { path: '/backtest', label: 'Backtest', icon: FlaskConical },
      { path: '/mf-tracker', label: 'MF Tracker', icon: PieChart },
      { path: '/monitoring', label: 'Monitoring', icon: Heart },
      { path: '/pipeline', label: 'Pipeline', icon: RefreshCcw },
    ],
  },
]

function LayoutDashboard(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" {...props}>
      <rect width="7" height="9" x="3" y="3" rx="1"/><rect width="7" height="5" x="14" y="3" rx="1"/><rect width="7" height="9" x="14" y="12" rx="1"/><rect width="7" height="5" x="3" y="16" rx="1"/>
    </svg>
  )
}

interface SidebarProps {
  collapsed: boolean
  onToggle: () => void
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const { theme, toggle } = useTheme()
  const { logout } = useAuth()

  return (
    <aside className={`fixed left-0 top-0 z-40 h-screen border-r border-border bg-card/90 backdrop-blur-xl transition-all duration-300 ${collapsed ? 'w-14' : 'w-52'}`}>
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-12 items-center justify-between border-b border-border px-2.5">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <Zap className="h-4 w-4 text-cyan" />
              <span className="font-mono text-xs font-bold tracking-[0.25em] text-foreground">STOMAR</span>
            </div>
          )}
          {collapsed && <Zap className="h-4 w-4 text-cyan mx-auto" />}
          <button onClick={onToggle} className="rounded p-1 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors">
            {collapsed ? <ChevronRight className="h-3 w-3" /> : <ChevronLeft className="h-3 w-3" />}
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-2 px-1.5">
          {navSections.map((section) => (
            <div key={section.title} className="mb-3">
              {!collapsed && (
                <p className="px-2 py-1 text-[9px] uppercase font-mono font-bold tracking-[0.2em] text-muted-foreground/50">
                  {section.title}
                </p>
              )}
              {section.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center gap-2 rounded-md px-2 py-1.5 text-xs transition-all duration-150 ${
                      isActive
                        ? 'bg-cyan/10 text-cyan font-medium border-l-2 border-cyan'
                        : 'text-muted-foreground hover:bg-accent/40 hover:text-foreground border-l-2 border-transparent'
                    } ${collapsed ? 'justify-center' : ''}`
                  }
                >
                  <item.icon className="h-3.5 w-3.5 shrink-0" />
                  {!collapsed && <span className="font-mono">{item.label}</span>}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-border p-1.5">
          <button
            onClick={toggle}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <span className="text-sm">{theme === 'dark' ? '☀' : '☽'}</span>
            {!collapsed && <span className="font-mono">{theme === 'dark' ? 'Light' : 'Dark'}</span>}
          </button>
          <button
            onClick={logout}
            className="flex w-full items-center gap-2 rounded-md px-2 py-1.5 text-xs text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <LogOut className="h-3.5 w-3.5 shrink-0" />
            {!collapsed && <span className="font-mono">Sign out</span>}
          </button>
        </div>
      </div>
    </aside>
  )
}

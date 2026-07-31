import React from 'react'
import { NavLink } from 'react-router'
import {
  Activity, BarChart3, Brain, Briefcase,
  ChevronLeft, ChevronRight, Coins, FileText,
  FlaskConical, Gauge, Heart, Layers,
  Network, PieChart, RefreshCcw,
  ScanSearch, Shield, SlidersHorizontal, Sparkles, Target,
  Thermometer, Zap, BookOpen
} from 'lucide-react'
import { useTheme } from './ThemeProvider'

const navSections = [
  {
    title: 'Core',
    items: [
      { path: '/', label: 'Dashboard', icon: LayoutDashboard },
      { path: '/wealth-goals', label: 'Wealth Goals', icon: Coins },
      { path: '/portfolio', label: 'Portfolio', icon: Briefcase },
      { path: '/paper-trading', label: 'Paper Trading', icon: FileText },
      { path: '/ledger', label: 'Ledger', icon: BookOpen },
    ],
  },
  {
    title: 'Intelligence',
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
    title: 'Risk & Analytics',
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
    title: 'Operations',
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

  return (
    <aside className={`fixed left-0 top-0 z-40 h-screen border-r border-border bg-card/80 backdrop-blur-xl transition-all duration-300 ${collapsed ? 'w-16' : 'w-56'}`}>
      <div className="flex h-full flex-col">
        {/* Logo */}
        <div className="flex h-14 items-center justify-between border-b border-border px-3">
          {!collapsed && (
            <div className="flex items-center gap-2">
              <Zap className="h-5 w-5 text-cyan" />
              <span className="font-mono text-sm font-bold tracking-wider text-foreground">STOMAR</span>
            </div>
          )}
          <button onClick={onToggle} className="rounded-md p-1.5 text-muted-foreground hover:bg-accent hover:text-foreground transition-colors">
            {collapsed ? <ChevronRight className="h-4 w-4" /> : <ChevronLeft className="h-4 w-4" />}
          </button>
        </div>

        {/* Nav */}
        <nav className="flex-1 overflow-y-auto py-2 px-2">
          {navSections.map((section) => (
            <div key={section.title} className="mb-3">
              {!collapsed && (
                <p className="px-2.5 py-1 text-[10px] uppercase font-mono font-semibold tracking-wider text-muted-foreground/70">
                  {section.title}
                </p>
              )}
              {section.items.map((item) => (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition-all duration-150 ${
                      isActive
                        ? 'bg-accent text-cyan font-medium'
                        : 'text-muted-foreground hover:bg-accent/50 hover:text-foreground'
                    } ${collapsed ? 'justify-center' : ''}`
                  }
                >
                  <item.icon className="h-4 w-4 shrink-0" />
                  {!collapsed && <span>{item.label}</span>}
                </NavLink>
              ))}
            </div>
          ))}
        </nav>

        {/* Footer */}
        <div className="border-t border-border p-2">
          <button
            onClick={toggle}
            className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-sm text-muted-foreground hover:bg-accent hover:text-foreground transition-colors"
          >
            <span className="text-base">{theme === 'dark' ? '☀️' : '🌙'}</span>
            {!collapsed && <span>{theme === 'dark' ? 'Light' : 'Dark'} Mode</span>}
          </button>
        </div>
      </div>
    </aside>
  )
}

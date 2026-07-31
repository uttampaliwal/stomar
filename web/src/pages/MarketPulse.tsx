import { Card, Stat, SectionHeader, Spinner, ErrorDisplay, EmptyState, PageHeader } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import type { MarketPulseResponse } from '../lib/api-types'

type TimeframeInfo = { direction: number; strength: number; signal: string; indicators: Record<string, string> }
type OptionsPcrData = NonNullable<MarketPulseResponse['options_pcr']> & { pcr?: number }

export default function MarketPulse() {
  const { data, loading, error } = useApi<MarketPulseResponse & { timeframes?: Record<string, TimeframeInfo> }>('/api/market/pulse')

  return (
    <div className="space-y-6">
      <PageHeader
        title="Market Pulse"
        description="FII/DII flows, options PCR, and multi-timeframe signals"
        badge="Context"
      />

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          This view brings together the broader market context that supports signal selection and risk stance.
        </p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !('error' in data) ? (
        <>
          {/* FII/DII */}
          {data.fii_dii && (
            <>
              <SectionHeader title="Institutional Flows" />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <Stat
                  label="FII Net"
                  value={`₹${Math.abs(data.fii_dii.fii_net).toFixed(0)} Cr`}
                  sub={data.fii_dii.fii_net >= 0 ? 'Inflow' : 'Outflow'}
                  trend={data.fii_dii.fii_net >= 0 ? 'up' : 'down'}
                />
                <Stat
                  label="DII Net"
                  value={`₹${Math.abs(data.fii_dii.dii_net).toFixed(0)} Cr`}
                  sub={data.fii_dii.dii_net >= 0 ? 'Inflow' : 'Outflow'}
                  trend={data.fii_dii.dii_net >= 0 ? 'up' : 'down'}
                />
                <Stat
                  label="Flow Sentiment"
                  value={data.fii_dii.flow_sentiment}
                  trend={data.fii_dii.flow_sentiment === 'Bullish' ? 'up' : 'down'}
                />
              </div>
            </>
          )}

          {/* Options PCR */}
          {data.options_pcr && (
            <>
              <SectionHeader title="Options PCR" />
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                <Stat label="PCR" value={(data.options_pcr as OptionsPcrData).pcr?.toFixed(3) || '—'} />
                <Stat label="Call OI" value={data.options_pcr.call_oi?.toLocaleString() || '—'} />
                <Stat label="Put OI" value={data.options_pcr.put_oi?.toLocaleString() || '—'} />
              </div>
            </>
          )}

          {/* Multi-Timeframe */}
          {data.multi_timeframe && (
            <>
              <SectionHeader title="Multi-Timeframe Signal" />
              <Card>
                <div className="text-center py-4">
                  <p className="font-mono text-3xl font-bold">{data.multi_timeframe.signal || '—'}</p>
                  <p className="text-sm text-muted-foreground mt-1">Confidence: {((data.multi_timeframe.confidence || 0) * 100).toFixed(1)}%</p>
                </div>
              </Card>
            </>
          )}

          {/* Timeframe Breakdown */}
          {data.timeframes && Object.keys(data.timeframes).length > 0 && (
            <>
              <SectionHeader title="Timeframe Breakdown" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(data.timeframes).map(([tf, info]) => (
                  <Card key={tf}>
                    <p className="text-xs font-semibold uppercase tracking-wider text-muted-foreground mb-1">{tf}</p>
                    <p className="font-mono text-lg font-bold">{info.signal || '—'}</p>
                    <p className="text-xs text-muted-foreground">Strength: {info.strength?.toFixed(2) || '—'}</p>
                  </Card>
                ))}
              </div>
            </>
          )}
        </>
      ) : (
        !loading && <EmptyState message="No market pulse data is available right now." />
      )}
    </div>
  )
}

import { useState } from 'react'
import { Card, SectionHeader, Spinner, ErrorDisplay, Badge } from '@/components/UI'
import { useApi } from '@/hooks/useApi'
import { useDebouncedValue } from '@/hooks/useDebouncedValue'

export default function Sentiment() {
  const [ticker, setTicker] = useState('RELIANCE.NS')
  const debouncedTicker = useDebouncedValue(ticker, 400)
  const { data, loading, error } = useApi<any>(`/api/sentiment/${debouncedTicker}`)
  const stocks = useApi<{ stocks: string[] }>('/api/market/stocks')

  const score = data?.score ?? 0
  const label = data?.label || 'Neutral'
  const headlines = data?.headlines || []
  const sourceScores = data?.source_scores || {}

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">News Sentiment</h1>
          <p className="text-sm text-muted-foreground">AI-powered sentiment analysis from financial news</p>
        </div>
        <select
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          className="rounded-lg border border-border bg-card px-3 py-2 text-sm font-mono focus:outline-none focus:ring-1 focus:ring-cyan"
        >
          {stocks.data?.stocks?.map((s: string) => (
            <option key={s} value={s}>{s}</option>
          ))}
        </select>
      </div>

      <Card className="border-l-4 border-l-cyan bg-cyan/5">
        <p className="text-sm text-muted-foreground">
          <span className="font-semibold text-foreground">AI-powered sentiment scoring</span> for each stock.
          Scores range from -1 (very negative) to +1 (very positive).
          This is one input to the Consensus engine.
        </p>
      </Card>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && !data.error && (
        <>
          <Card className="text-center py-6">
            <Badge variant={
              label === 'Bullish' || label === 'Positive' ? 'success' :
              label === 'Bearish' || label === 'Negative' ? 'danger' :
              'warning'
            } className="text-lg px-6 py-1">
              {label === 'Bullish' || label === 'Positive' ? '▲ BULLISH' :
               label === 'Bearish' || label === 'Negative' ? '▼ BEARISH' :
               '● NEUTRAL'}
            </Badge>
            <p className="text-sm text-muted-foreground mt-2">
              Score: <span className={`font-mono font-bold ${score >= 0 ? 'text-emerald' : 'text-rose'}`}>{score.toFixed(3)}</span>
            </p>
          </Card>

          {/* Score Bar */}
          <Card>
            <p className="text-xs text-muted-foreground uppercase mb-2">Sentiment Score</p>
            <div className="relative h-4 bg-secondary rounded-full overflow-hidden">
              <div className="absolute left-1/2 top-0 w-px h-full bg-muted-foreground/30" />
              <div
                className={`absolute top-0 h-full rounded-full ${score >= 0 ? 'bg-emerald' : 'bg-rose'}`}
                style={{
                  left: score >= 0 ? '50%' : `${50 + score * 50}%`,
                  width: `${Math.abs(score) * 50}%`,
                }}
              />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground mt-1">
              <span>-1.0</span>
              <span>0</span>
              <span>+1.0</span>
            </div>
          </Card>

          {Object.keys(sourceScores).length > 0 && (
            <>
              <SectionHeader title="Source Scores" />
              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                {Object.entries(sourceScores).map(([source, srcScore]) => (
                  <Card key={source}>
                    <p className="text-xs text-muted-foreground uppercase">{source}</p>
                    <p className={`font-mono text-xl font-bold ${(srcScore as number) >= 0 ? 'text-emerald' : 'text-rose'}`}>
                      {(srcScore as number).toFixed(3)}
                    </p>
                    <div className="w-full bg-secondary rounded-full h-1.5 mt-2">
                      <div
                        className={`h-1.5 rounded-full ${(srcScore as number) >= 0 ? 'bg-emerald' : 'bg-rose'}`}
                        style={{ width: `${Math.abs(srcScore as number) * 100}%` }}
                      />
                    </div>
                  </Card>
                ))}
              </div>
            </>
          )}

          {headlines.length > 0 && (
            <>
              <SectionHeader title="Recent Headlines" />
              <div className="space-y-2">
                {headlines.map((item: any, i: number) => (
                  <Card key={i} className="hover:ring-1 hover:ring-cyan/50 transition-all">
                    <div className="flex justify-between items-start gap-3">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium line-clamp-2">{typeof item === 'string' ? item : item.headline || item.title || ''}</p>
                        {typeof item !== 'string' && item.source && (
                          <p className="text-xs text-muted-foreground mt-1">{item.source}</p>
                        )}
                      </div>
                      {typeof item !== 'string' && item.sentiment !== undefined && (
                        <Badge variant={
                          item.sentiment > 0.2 ? 'success' :
                          item.sentiment < -0.2 ? 'danger' :
                          'default'
                        }>
                          {(item.sentiment * 100).toFixed(0)}%
                        </Badge>
                      )}
                    </div>
                  </Card>
                ))}
              </div>
            </>
          )}

          {headlines.length === 0 && Object.keys(sourceScores).length === 0 && (
            <Card className="text-center py-6">
              <p className="text-sm text-muted-foreground">No sentiment data available for this stock</p>
            </Card>
          )}
        </>
      )}
    </div>
  )
}

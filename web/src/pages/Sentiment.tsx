import { Card, Stat, SectionHeader, Spinner, ErrorDisplay } from '@/components/UI'
import { useApi } from '@/hooks/useApi'

export default function Sentiment() {
  const { data, loading, error } = useApi<any>('/api/sentiment/RELIANCE.NS')

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight">Sentiment</h1>
        <p className="text-sm text-muted-foreground">FinBERT-powered multi-source sentiment</p>
      </div>

      {loading && <Spinner />}
      {error && <ErrorDisplay message={error} />}

      {data && (
        <>
          <Card className="text-center py-6">
            <p className="text-xs uppercase tracking-wider text-muted-foreground mb-2">Sentiment Score</p>
            <p className={`font-mono text-4xl font-bold ${data.score > 0 ? 'text-emerald' : data.score < 0 ? 'text-rose' : 'text-muted-foreground'}`}>
              {data.score?.toFixed(3) || '0'}
            </p>
            <p className="text-sm mt-2">{data.label || 'Neutral'}</p>
          </Card>

          {data.headlines?.length > 0 && (
            <>
              <SectionHeader title="Recent Headlines" />
              <div className="space-y-2">
                {data.headlines.map((h: any, i: number) => (
                  <Card key={i}>
                    <p className="text-sm">{h.title || h}</p>
                    {h.source && <p className="text-xs text-muted-foreground mt-1">{h.source}</p>}
                  </Card>
                ))}
              </div>
            </>
          )}
        </>
      )}
    </div>
  )
}

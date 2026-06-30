import './App.css'

const metrics = [
  { label: 'Consensus', value: 'BUY', detail: 'Reliance • 82% confidence', tone: 'positive' },
  { label: 'Risk', value: 'Moderate', detail: 'VaR 1.8% • Kelly 0.31', tone: 'neutral' },
  { label: 'P&L', value: '+₹18.4k', detail: 'Paper account • +3.2%', tone: 'positive' },
]

const watchlist = [
  { symbol: 'RELIANCE', signal: 'Strong Bull', confidence: '82%', note: 'Momentum + regime alignment' },
  { symbol: 'TCS', signal: 'Bullish', confidence: '71%', note: 'Sentiment improved after earnings' },
  { symbol: 'HDFCBANK', signal: 'Watch', confidence: '54%', note: 'Volatility expanded, risk budget tight' },
]

const workflow = [
  { step: 'Signal intake', detail: '14 modules feed the meta-controller' },
  { step: 'Risk gating', detail: 'VaR, drawdown, and Kelly constraints' },
  { step: 'Execution', detail: 'Paper or live order routing ready' },
]

function App() {
  return (
    <div className="app-shell">
      <header className="hero-card">
        <div className="hero-copy">
          <p className="eyebrow">StoMar command center</p>
          <h1>Turn signal density into decisive action.</h1>
          <p>
            A focused terminal for market pulse, ensemble confidence, risk posture, and execution readiness — designed for quantified trading workflows.
          </p>
          <div className="hero-actions">
            <button className="primary">Open signal board</button>
            <button className="secondary">Review risk stack</button>
          </div>
        </div>

        <div className="hero-panel">
          <div className="panel-chip">Live market pulse • 08:42 IST</div>
          <div className="panel-metric">
            <span>+3.1%</span>
            <small>model edge this session</small>
          </div>
          <div className="panel-bars" aria-hidden="true">
            <span></span>
            <span></span>
            <span></span>
            <span></span>
          </div>
        </div>
      </header>

      <section className="stats-grid" aria-label="Core trading metrics">
        {metrics.map((item) => (
          <article key={item.label} className={`stat-card ${item.tone}`}>
            <p>{item.label}</p>
            <strong>{item.value}</strong>
            <small>{item.detail}</small>
          </article>
        ))}
      </section>

      <section className="content-grid">
        <article className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Watchlist</p>
              <h2>Highest-conviction signals</h2>
            </div>
            <button className="ghost">View screener</button>
          </div>

          <ul className="stream-list">
            {watchlist.map((item) => (
              <li key={item.symbol}>
                <div>
                  <h3>{item.symbol}</h3>
                  <p>{item.note}</p>
                </div>
                <div className="signal-pill">
                  <span>{item.signal}</span>
                  <strong>{item.confidence}</strong>
                </div>
              </li>
            ))}
          </ul>
        </article>

        <aside className="panel side-panel">
          <div className="panel-heading compact">
            <div>
              <p className="eyebrow">Workflow</p>
              <h2>How the system thinks</h2>
            </div>
          </div>

          {workflow.map((item) => (
            <div className="mini-metric" key={item.step}>
              <span>{item.step}</span>
              <p>{item.detail}</p>
            </div>
          ))}
        </aside>
      </section>
    </div>
  )
}

export default App

// Typed response shapes for StoMar API endpoints.
// Mirrors the JSON emitted by api/routers/*.py — keep in sync when the backend changes.

export interface ApiError {
  error: string
}

export interface Candle {
  date: string
  open: number
  high: number
  low: number
  close: number
  volume: number
}

// ── /api/backtest/{ticker} ──
export interface BacktestWindow {
  window: string
  ensemble_accuracy: number
  total_return: number
  sharpe: number
  total_trades: number
}

export interface BacktestResponse {
  ticker: string
  ensemble_accuracy: number
  annual_return: number
  sharpe: number
  sortino: number
  calmar: number
  max_drawdown: number
  volatility: number
  var_95: number
  cvar_95: number
  total_return: number
  total_trades: number
  win_rate: number
  profit_factor: number
  test_results?: BacktestWindow[]
}

// ── /api/consensus/ ──
export interface ConsensusResult {
  ticker: string
  ensemble_signal: string
  ensemble_confidence: number
  meta_signal: string
  meta_confidence: number
  regime: string
  consensus: string
}

export interface ConsensusResponse {
  results: ConsensusResult[]
  buy_count: number
  sell_count: number
  hold_count: number
  conflicted: number
  scanned: number
  total: number
}

// ── /api/correlation/ ──
export interface CorrelationResponse {
  tickers: string[]
  matrix: number[][]
}

// ── /api/pipeline/status ──
export interface PipelineStatusResponse {
  total: number
  trained: number
  missing: number
  trained_tickers: string[]
  missing_tickers: string[]
}

// ── /api/monitoring/ ──
export interface MonitoringAlert {
  severity: string
  message: string
}

export interface MonitoringResponse {
  models_trained: string[]
  models_missing: string[]
  alerts: {
    critical: MonitoringAlert[]
    warning: MonitoringAlert[]
    info: MonitoringAlert[]
  }
  alert_history: Record<string, unknown>[]
  pipeline_failures_last_24h: number
}

// ── /api/insights ──
export interface RecommendationResponse {
  ticker: string
  action: string
  confidence: number
  rationale: string
  risk_level: string
  price: number
  trend_pct: number
  volatility: number
}

export interface EnrichmentResponse {
  ticker: string
  price: number | null
  change_pct: number
  sentiment_score: number
  sentiment_label: string
  headline_count: number
  headlines: { title: string; source: string }[]
}

// ── /api/automation ──
export interface LedgerDecision {
  id: number
  date: string
  ticker: string
  ensemble_direction: number | null
  ensemble_confidence: number | null
  sentiment_score: number | null
  fii_net: number | null
  dii_net: number | null
  pcr: number | null
  max_pain: number | null
  mtf_signal: number | null
  regime: string | null
  regime_confidence: number | null
  var_95: number | null
  cvar_95: number | null
  sharpe: number | null
  volatility_forecast: number | null
  fundamental_score: number | null
  action: string
  position_size: number | null
  confidence: number | null
  reasoning: string | null
  actual_return: number | null
  actual_direction: number | null
  correct: number | null
  created_at: string
}

export interface AutomationDecisionsResponse {
  decisions: LedgerDecision[]
}

export interface AutomationRunResponse {
  ok: boolean
  summary: {
    date: string
    decisions: {
      ticker: string
      action: string
      position_size: number
      confidence: number
      reasoning: string
      current_price: number
      decision_id?: number
    }[]
    trades: {
      ticker: string
      side: string
      quantity: number
      price: number
      position_size: number
      stop_price: number
    }[]
    errors: { ticker: string; error: string }[]
  }
}

// ── /api/paper-trading ──
export interface PaperPositionSummary {
  initial_capital: number
  current_equity: number
  cash: number
  total_return_pct: number
  total_trades: number
  open_positions: Record<string, {
    quantity: number
    avg_cost: number
    current_price: number
    market_value: number
    unrealized_pnl: number
    unrealized_pnl_pct: number
  }>
  closed_positions: number
  realized_pnl: number
  unrealized_pnl: number
  win_rate: number
  risk_status: {
    halted: boolean
    halt_reason: string
    current_equity: number
    peak_equity: number
    drawdown_pct: number
    daily_pnl: number
    weekly_pnl: number
    daily_loss_remaining: number
    weekly_loss_remaining: number
    consecutive_losses: number
  }
}

export interface PaperPosition {
  ticker: string
  quantity: number
  avg_cost: number
  current_price: number
  market_value: number
  pnl: number
  pnl_pct: number
}

export interface PaperPositionsResponse {
  positions: PaperPosition[]
}

export interface PaperTrade {
  ticker: string
  side: string
  quantity: number
  price: number
  fill_cost: number
  pnl: number
  date: string
}

export interface PaperTradesResponse {
  trades: PaperTrade[]
}

export interface PaperOrderResponse {
  order_id: string | null
  status: string
  price: number
  filled_quantity?: number
  filled_price?: number
  summary?: PaperPositionSummary
}

export interface PaperCloseResponse {
  status: string
  ticker: string
  price: number
  pnl: number
  summary: PaperPositionSummary
}

export interface PaperResetResponse {
  status: string
  message: string
  state: PaperPositionSummary
}

// ── /api/ledger ──
export interface LedgerSnapshot {
  id: number
  date: string
  total_value: number
  cash: number
  holdings_json: string
  created_at: string
}

export interface LedgerSummaryResponse {
  total_decisions: number
  total_trades: number
  accuracy: number
  trade_accuracy?: number | null
  hold_ratio?: number | null
  signal_accuracy: Record<string, number | null>
  snapshots: LedgerSnapshot[]
}

export interface LedgerDecisionsResponse {
  decisions: LedgerDecision[]
}

export interface BenchmarkSeries {
  dates: string[]
  cumulative: number[]
}

export interface LedgerBenchmarkResponse {
  start_date?: string
  end_date?: string
  paper?: BenchmarkSeries
  nifty?: BenchmarkSeries
  paper_total_return?: number
  nifty_total_return?: number
  alpha?: number
  information_ratio?: number | null
  paper_max_drawdown?: number
  nifty_max_drawdown?: number
  n_snapshots?: number
  nifty_points?: number
  error?: string
}

export interface CalibrationBin {
  bin: string
  n: number
  mean_confidence: number | null
  empirical_accuracy: number | null
}

export interface CalibrationResponse {
  n: number
  overall_accuracy?: number | null
  mean_confidence?: number | null
  bins?: CalibrationBin[]
  ece?: number | null
  mce?: number | null
  note?: string
  error?: string
}

export interface ReadinessGate {
  passed: boolean
  detail: string
}

export interface ReadinessResponse {
  ready: boolean
  gates?: Record<string, ReadinessGate>
  evidence?: Record<string, unknown>
  summary?: string
  checked_at?: string
  error?: string
}

// ── /api/mf-tracker/portfolio ──
export interface MfHolding {
  ticker: string
  fund_name: string
  units: number
  avg_nav: number
  current_value: number
  pnl: number
  return_pct: number
}

export interface MfPortfolioResponse {
  invested: number
  current_value: number
  pnl: number
  return_pct: number
  holdings: MfHolding[]
  allocation: Record<string, number>
  concentration_risk: {
    ticker: string
    fund_name: string
    weight: number
    threshold: number
  }[]
}

// ── /api/market/pulse ──
export interface MarketPulseResponse {
  fii_dii?: {
    fii_net: number
    dii_net: number
    flow_sentiment: string
  }
  options_pcr?: {
    pcr_oi: number
    pcr_volume: number
    max_pain: number
    call_oi: number
    put_oi: number
    symbol: string
  }
  multi_timeframe?: {
    direction: number
    signal: string
    confidence: number
    weighted_score: number
    timeframes: Record<string, {
      direction: number
      strength: number
      signal: string
      indicators: Record<string, string>
    }>
  }
}

// ── /api/optimizer/ ──
export interface OptimizerResponse {
  tickers: string[]
  max_sharpe: {
    weights: number[]
    return: number
    volatility: number
    sharpe: number
    tickers: string[]
  }
  min_variance: {
    weights: number[]
    return: number
    volatility: number
    sharpe: number
    tickers: string[]
  }
  black_litterman: {
    weights: number[]
    expected_return: number[]
    posterior_return: number[]
    tickers: string[]
  }
  efficient_frontier: {
    return: number
    volatility: number
    weights: number[]
  }[]
  returns_stats: {
    mean_daily: Record<string, number>
    annualized: Record<string, number>
  }
  correlation: Record<string, Record<string, number>>
}

// ── /api/portfolio ──
export interface PortfolioPosition {
  ticker: string
  quantity: number
  avg_cost: number
  current_price: number
  pnl: number
  pnl_pct: number
}

export interface PortfolioStatsResponse {
  equity: number
  cash: number
  positions: PortfolioPosition[]
  realized_pnl: number
  unrealized_pnl: number
  total_trades: number
  sharpe_ratio: number
  win_rate: number
  closed_positions: number
}

export interface PortfolioTradesResponse {
  trades: {
    ticker: string
    side: string
    quantity: number
    price: number
    date: string
  }[]
}

// ── /api/predictions ──
export interface PredictionResponse {
  ticker: string
  metrics: {
    price: number
    day_change: number
    day_change_pct: number
    volume: number
    atr: number
    rsi: number
    macd: number
  }
  prediction: null | {
    direction: string
    confidence: number
    conviction: number
    details: Record<string, number | string>
  }
  recent: Candle[]
  chart: Candle[]
}

export interface FeatureImportanceResponse {
  features: {
    name: string
    importance: number
  }[]
}

export interface TrainResponse {
  status: string
  message?: string
  error?: string
  xgb_accuracy: number
  lgb_accuracy: number
  lstm_accuracy: number
  gru_accuracy: number
  transformer_accuracy: number
  ensemble_accuracy: number
}

// ── /api/ranking/ ──
export interface RankingResult {
  ticker: string
  composite_score: number
  momentum: {
    mom_5d: number
    mom_20d: number
    mom_60d: number
    mom_120d: number
    momentum_combined: number
  }
  volatility: {
    volatility: number
    vol_score: number
  }
  volume: {
    relative_volume: number
    volume_score: number
  }
  technical: {
    rsi: number
    rsi_score: number
    macd: number
    macd_signal: number
    macd_score: number
    price_vs_sma20: number
    price_vs_sma50: number
    sma_score: number
    bb_position: number
    bb_score: number
    technical_combined: number
  }
  ml_score: number
  fundamental_score: number
  fundamentals: Record<string, never>
  current_price: number
  daily_return: number
  rank: number
  percentile: number
}

export interface RankingResponse {
  rankings: RankingResult[]
}

// ── /api/regime ──
export interface RegimeRecommendation {
  action: string
  allocation: string
  risk_level: string
}

export interface RegimeResponse {
  ticker: string
  regime: string
  confidence: number
  bull_signals: number
  bear_signals: number
  total_signals: number
  indicators: Record<string, string>
  recommendation: RegimeRecommendation
}

// ── /api/risk ──
export interface RiskResponse {
  ticker: string
  report: Record<string, number>
  regime: Omit<RegimeResponse, 'ticker'>
  quant_stats: Record<string, number>
  kelly: {
    optimal_fraction: number
    win_rate: number
    avg_win: number
    avg_loss: number
  }
}

// ── /api/scanner/ ──
export interface ScannerResult {
  ticker: string
  signal: string
  confidence: number
  price: number
  day_return: number
  chg_5d: number
  rsi: number
}

export interface ScannerResponse {
  results: ScannerResult[]
  buy_count: number
  sell_count: number
  scanned: number
  total: number
  trained: number
}

// ── /api/scenarios ──
export interface ScenarioResult {
  name: string
  total_return: number
  annualized_return: number
  annualized_vol: number
  sharpe: number
  max_drawdown: number
  n_trades: number
}

export interface ScenariosResponse {
  ticker: string
  scenarios: ScenarioResult[]
}

// ── /api/sentiment ──
export interface SentimentResponse {
  ticker: string
  score: number
  label: string
  headlines: unknown[]
  source_scores: Record<string, unknown>
}

// ── /api/volatility ──
export interface VolatilityResponse {
  ticker: string
  current_vol: number
  regime: string
  percentile: number
  historical_vol: number
  ewma_vol: number
  parkinson_vol: number
  garman_klass_vol: number
  yang_zhang_vol: number
  atr_pct: number
  bb_width: number
  bb_pct_b: number
  forecast: {
    horizon: number
    forecast_vols: number[]
    current_vol: number
    long_term_vol: number
  }
  position_sizing: {
    recommended_size: number
    vol_scalar: number
    reasoning: string
  }
}

// ── /api/wealth ──
export interface WealthStrategy {
  name: string
  description: string
  cagr: string
  max_drawdown: string
  sharpe: string
  complexity: string
}

export interface WealthStrategiesResponse {
  strategies: WealthStrategy[]
}

export interface MonteCarloResponse {
  probability: number
  median_corpus: number
  mean_corpus: number
  real_corpus: number
  real_p10: number
  real_p50: number
  real_p90: number
  target: number
  shortfall_risk: number
  recommended_sip: number
  sip_topup: number
  worst_5pct: number
  best_5pct: number
  var_95: number
  cvar_95: number
  percentiles: {
    p5: number[]
    p10: number[]
    p25: number[]
    p50: number[]
    p75: number[]
    p90: number[]
    p95: number[]
    labels: string[]
  }
  params: {
    current_capital: number
    monthly_sip: number
    step_up_pct: number
    horizon_years: number
    expected_return: number
    volatility: number
    inflation_rate: number
    simulations: number
    t_df: number
  }
}

export interface AdvisorResponse {
  source: string
  advisor: {
    executive_summary: string
    asset_allocation: {
      equity_pct: number
      debt_pct: number
      gold_pct: number
      cash_pct: number
    }
    recommended_funds: string[]
    action_items: string[]
    risk_warnings: string[]
    expected_cagr: string
    expected_timeline: string
  }
  note?: string
}

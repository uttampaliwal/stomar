import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os, sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import fetch_stock_data, NSE_STOCKS, get_market_status
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.trainer import train_for_ticker, FEATURE_COLS
from src.ensemble import predict_ensemble
from src.portfolio import Portfolio
from src.backtester import run_walk_forward_backtest
from src.sentiment import get_stock_sentiment
from src.flow import get_flow_sentiment, fetch_options_pcr
from src.multitimeframe import fetch_mtf_data, get_combined_signal
from src.risk import generate_risk_report, kelly_criterion
from src.optimizer import optimize_portfolio
from src.holdings import parse_holdings_csv, compute_portfolio_stats
from src.regime import detect_regime

st.set_page_config(page_title="StoMar | Quant Intelligence", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown("""
<style>
/* ═══════════════════════════════════════════════════════════════
   PROFESSIONAL QUANT TERMINAL — Dark-First Design System
   ═══════════════════════════════════════════════════════════════ */

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --bg-card-hover: rgba(30, 41, 59, 0.9);
    --bg-card-glass: rgba(15, 23, 42, 0.6);
    --border-primary: rgba(51, 65, 85, 0.5);
    --border-hover: rgba(99, 179, 237, 0.4);
    --border-glow: rgba(99, 179, 237, 0.2);
    --accent-cyan: #22d3ee;
    --accent-blue: #3b82f6;
    --accent-violet: #8b5cf6;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --accent-slate: #94a3b8;
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --text-dim: rgba(148, 163, 184, 0.6);
    --gradient-brand: linear-gradient(135deg, #22d3ee 0%, #3b82f6 50%, #8b5cf6 100%);
    --gradient-profit: linear-gradient(135deg, #10b981, #22d3ee);
    --gradient-loss: linear-gradient(135deg, #f43f5e, #f59e0b);
    --gradient-card: linear-gradient(135deg, rgba(34, 211, 238, 0.03), rgba(59, 130, 246, 0.03));
    --shadow-sm: 0 1px 2px rgba(0,0,0,0.3);
    --shadow-md: 0 4px 12px rgba(0,0,0,0.4);
    --shadow-lg: 0 8px 32px rgba(0,0,0,0.5);
    --shadow-glow: 0 0 40px rgba(34, 211, 238, 0.08);
    --radius-sm: 8px;
    --radius-md: 12px;
    --radius-lg: 16px;
    --radius-xl: 24px;
    --font-sans: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'JetBrains Mono', 'Fira Code', monospace;
}

/* ─── Global Reset ─── */
.stApp {
    font-family: var(--font-sans) !important;
    background: var(--bg-primary) !important;
    color: var(--text-primary) !important;
}
.stApp > header { background: transparent !important; }

/* ─── Typography ─── */
h1, h2, h3, h4, h5, h6, p, span, div, label {
    font-family: var(--font-sans) !important;
}

/* ─── Card System ─── */
.glass {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-lg);
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
}
.glass::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(34, 211, 238, 0.15), transparent);
    opacity: 0;
    transition: opacity 0.25s;
}
.glass:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-1px);
}
.glass:hover::before { opacity: 1; }

.glass-compact {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    padding: 1rem 1.2rem;
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}
.glass-compact:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
}

/* ─── Gradient Text ─── */
.gradient-text {
    background: var(--gradient-brand);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
    letter-spacing: -0.02em;
}

/* ─── Metric Cards ─── */
.metric-val {
    font-size: 1.75rem;
    font-weight: 800;
    font-family: var(--font-mono) !important;
    letter-spacing: -0.03em;
    line-height: 1.2;
    color: var(--text-primary);
}
.metric-val.profit { background: var(--gradient-profit); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
.metric-val.loss { background: var(--gradient-loss); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }

.metric-label {
    font-size: 0.7rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 0.3rem;
}

.metric-sub {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-secondary);
    font-family: var(--font-mono) !important;
}

/* ─── Tags & Badges ─── */
.tag {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    padding: 0.25rem 0.75rem;
    border-radius: 100px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    text-transform: uppercase;
}
.tag-buy { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
.tag-sell { background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.25); }
.tag-neutral { background: rgba(148, 163, 184, 0.08); color: var(--text-secondary); border: 1px solid var(--border-primary); }
.tag-up { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
.tag-down { background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.25); }
.tag-warn { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }
.tag-info { background: rgba(59, 130, 246, 0.12); color: #60a5fa; border: 1px solid rgba(59, 130, 246, 0.25); }

/* ─── Prediction Pill ─── */
.pred-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.5rem 1.5rem;
    border-radius: 100px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.02em;
}
.pred-pill.up { background: rgba(16, 185, 129, 0.1); color: #34d399; border: 2px solid rgba(16, 185, 129, 0.3); box-shadow: 0 0 20px rgba(16, 185, 129, 0.1); }
.pred-pill.down { background: rgba(244, 63, 94, 0.1); color: #fb7185; border: 2px solid rgba(244, 63, 94, 0.3); box-shadow: 0 0 20px rgba(244, 63, 94, 0.1); }

/* ─── Section Headers ─── */
.section-header {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid var(--border-primary);
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-header::before {
    content: '';
    width: 3px;
    height: 12px;
    background: var(--gradient-brand);
    border-radius: 2px;
}

/* ─── Buttons ─── */
.stButton > button {
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.01em !important;
    border: 1px solid var(--border-primary) !important;
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    transition: all 0.2s ease !important;
    padding: 0.5rem 1.2rem !important;
}
.stButton > button:hover {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 16px rgba(34, 211, 238, 0.15) !important;
    background: var(--bg-card-hover) !important;
}
.stButton > button[kind="primary"] {
    background: var(--gradient-brand) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 0 24px rgba(34, 211, 238, 0.25) !important;
    transform: translateY(-1px);
}

/* ─── Tabs ─── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0.25rem;
    background: var(--bg-secondary);
    border-radius: var(--radius-md);
    padding: 0.25rem;
    border: 1px solid var(--border-primary);
}
.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 0.5rem 1rem !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
    color: var(--text-secondary) !important;
    border: none !important;
    background: transparent !important;
    transition: all 0.2s ease !important;
}
.stTabs [data-baseweb="tab"]:hover {
    color: var(--text-primary) !important;
    background: rgba(51, 65, 85, 0.3) !important;
}
.stTabs [aria-selected="true"] {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    font-weight: 600 !important;
    border: 1px solid var(--border-primary) !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ─── Data Tables ─── */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    overflow: hidden;
}
[data-testid="stDataFrame"] th {
    background: var(--bg-secondary) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.7rem !important;
    letter-spacing: 0.05em !important;
    color: var(--text-muted) !important;
}

/* ─── Inputs ─── */
.stSelectbox > div > div,
.stMultiSelect > div > div,
.stNumberInput > div > div,
.stSlider > div > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border-primary) !important;
}
.stSelectbox > div > div:focus-within,
.stMultiSelect > div > div:focus-within {
    border-color: var(--accent-cyan) !important;
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.2) !important;
}

/* ─── Expanders ─── */
.streamlit-expanderHeader {
    border-radius: var(--radius-md) !important;
    background: var(--bg-card) !important;
    border: 1px solid var(--border-primary) !important;
    font-weight: 500 !important;
}

/* ─── Dividers ─── */
hr {
    border: none !important;
    border-top: 1px solid var(--border-primary) !important;
    margin: 1rem 0 !important;
}

/* ─── Scrollbar ─── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: var(--border-primary); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ─── Animations ─── */
@keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
@keyframes pulse-glow { 0%, 100% { box-shadow: 0 0 20px rgba(34, 211, 238, 0.1); } 50% { box-shadow: 0 0 40px rgba(34, 211, 238, 0.2); } }
.animate-in { animation: fadeIn 0.4s ease-out; }
.glow-pulse { animation: pulse-glow 3s ease-in-out infinite; }

/* ─── Live Indicator ─── */
.live-dot {
    display: inline-block;
    width: 6px; height: 6px;
    border-radius: 50%;
    margin-right: 6px;
    animation: pulse 2s ease-in-out infinite;
}
.live-dot.open { background: #10b981; box-shadow: 0 0 8px rgba(16, 185, 129, 0.5); }
.live-dot.closed { background: #f43f5e; box-shadow: 0 0 8px rgba(244, 63, 94, 0.5); }
@keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }

/* ─── Stat Grid ─── */
.stat-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
    gap: 0.75rem;
}
.stat-item {
    background: var(--bg-card);
    border: 1px solid var(--border-primary);
    border-radius: var(--radius-md);
    padding: 1rem;
    text-align: center;
    transition: all 0.2s ease;
}
.stat-item:hover { border-color: var(--border-hover); box-shadow: var(--shadow-glow); }

/* ─── Portfolio Bar ─── */
.alloc-bar {
    display: flex;
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
    background: var(--bg-secondary);
    margin: 0.5rem 0;
}
.alloc-bar > div { transition: width 0.3s ease; }

/* ─── Regime Badge ─── */
.regime-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.02em;
}
.regime-bull { background: rgba(16, 185, 129, 0.12); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.25); }
.regime-bear { background: rgba(244, 63, 94, 0.12); color: #fb7185; border: 1px solid rgba(244, 63, 94, 0.25); }
.regime-neutral { background: rgba(245, 158, 11, 0.12); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.25); }

/* ─── Footer ─── */
.footer-note {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-align: center;
    padding: 1rem 0;
    border-top: 1px solid var(--border-primary);
    margin-top: 2rem;
}

/* ─── Hide Streamlit Defaults ─── */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stDeployButton { display: none; }

/* ─── Responsive ─── */
@media (max-width: 768px) {
    .stat-grid { grid-template-columns: repeat(2, 1fr); }
    .metric-val { font-size: 1.3rem; }
}
</style>
""", unsafe_allow_html=True)


if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio(initial_capital=100000)
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = NSE_STOCKS[0]


@st.cache_data(ttl=3600)
def get_data(ticker, period):
    return fetch_stock_data(ticker, period=period)


def metric_card(label, value, delta=None, help_text=None, icon=None):
    icon_html = f'<span style="font-size:1.2rem;margin-right:0.3rem;">{icon}</span>' if icon else ""
    if delta is not None:
        d_str = str(delta)
        is_up = "+" in d_str or (d_str.replace('.','').replace('-','').isdigit() and not d_str.startswith('-'))
        color = "#10b981" if is_up else "#f43f5e"
        arrow = "▲" if is_up else "▼"
        delta_html = f'<div style="font-size:0.75rem;font-weight:600;color:{color};margin-top:0.25rem;">{arrow} {d_str}</div>'
    else:
        delta_html = ""
    h = f'<div style="font-size:0.7rem;color:var(--text-muted);margin-top:0.2rem;">{help_text}</div>' if help_text else ""
    return f"""<div class="stat-item animate-in">
        <div class="metric-label">{icon_html}{label}</div>
        <div class="metric-val">{value}</div>
        {delta_html}{h}
    </div>"""


def pred_card(direction, confidence, details=None):
    is_up = direction == 1
    cls = "up" if is_up else "down"
    arrow = "▲" if is_up else "▼"
    label = "BUY" if is_up else "SELL"
    tag_cls = "tag-buy" if is_up else "tag-sell"
    return f"""
    <div class="glass animate-in" style="text-align:center;padding:2rem 1.5rem;">
        <div class="metric-label" style="margin-bottom:0.75rem;">ENSEMBLE SIGNAL</div>
        <div class="pred-pill {cls}" style="margin:0 auto 1rem;font-size:1.4rem;">
            {arrow} {label}
        </div>
        <div class="tag {tag_cls}" style="font-size:0.8rem;padding:0.35rem 1rem;">
            {confidence:.0f}% confidence
        </div>
    </div>
    """


def tab_predictions():
    # ─── Top Bar ───
    market = get_market_status()
    dot_cls = "open" if market == "Open" else "closed"
    st.markdown(f"""
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <div style="display:flex;align-items:center;gap:0.75rem;">
            <span style="font-size:1.8rem;font-weight:800;letter-spacing:-0.03em;" class="gradient-text">StoMar</span>
            <span style="font-size:0.7rem;color:var(--text-muted);font-weight:500;letter-spacing:0.08em;text-transform:uppercase;">Quant Intelligence Platform</span>
        </div>
        <div style="display:flex;align-items:center;gap:1.5rem;">
            <div style="display:flex;align-items:center;gap:0.4rem;">
                <span class="live-dot {dot_cls}"></span>
                <span style="font-size:0.75rem;font-weight:600;color:{'#10b981' if market=='Open' else '#f43f5e'};">Market {market}</span>
            </div>
            <div style="font-size:0.75rem;color:var(--text-muted);font-family:var(--font-mono);">{datetime.now().strftime('%d %b %Y • %H:%M')}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ─── Controls ───
    c1, c2, c3, c4 = st.columns([3, 1.2, 1, 1])
    with c1:
        ticker = st.selectbox("Stock", NSE_STOCKS, index=NSE_STOCKS.index(st.session_state.last_ticker), label_visibility="collapsed")
        st.session_state.last_ticker = ticker
    with c2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=3, label_visibility="collapsed")
    with c3:
        refresh = st.button("↻ Refresh", width='stretch')
    with c4:
        train_btn = st.button("⚡ Train Model", width='stretch', type="primary")

    try:
        df = get_data(ticker, period)
    except Exception as e:
        st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;border-color:rgba(244,63,94,0.3);">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">❌</div>
            <div style="font-weight:600;color:#f43f5e;">Data Fetch Failed</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">{e}</div>
        </div>""", unsafe_allow_html=True)
        return

    df_feat = add_technical_indicators(df, ticker=ticker)
    curr = df["close"].iloc[-1]
    prev_c = df["close"].iloc[-2]
    chg = curr - prev_c
    chg_pct = (chg / prev_c) * 100
    ticker_display = ticker.replace(".NS", "")

    # Train action
    if train_btn:
        with st.spinner(f"Training {ticker_display}..."):
            r = train_for_ticker(ticker, force_retrain=True)
            st.toast(f"Trained: Ensemble {r.get('ensemble_accuracy',0):.1%}")
            st.cache_data.clear()

    # Prediction
    ens_dir, conf, details = None, 0, None
    if models_exist(ticker):
        try:
            lstm, gru, transformer, xgb, scaler, feat, lgb = load_models(ticker)
            ens_dir, conf, details = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat, lgb_model=lgb)
        except Exception as e:
            st.markdown(f"""<div class="glass" style="text-align:center;padding:1.5rem;border-color:rgba(245,158,11,0.3);">
                <div style="font-weight:600;color:#fbbf24;">Model Load Warning</div>
                <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">{e}. Retrain this stock.</div>
            </div>""", unsafe_allow_html=True)

    # ─── Top Metrics Row ───
    sign = "+" if chg >= 0 else ""
    p_cls = "profit" if chg >= 0 else "loss"
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        st.markdown(metric_card("PRICE", f"₹{curr:,.2f}", f"{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)", icon="📈"), unsafe_allow_html=True)
    with r2:
        rng = f"₹{df['low'].iloc[-1]:,.0f} – ₹{df['high'].iloc[-1]:,.0f}"
        st.markdown(metric_card("DAY RANGE", rng, help_text="Low – High", icon="📊"), unsafe_allow_html=True)
    with r3:
        vol = f"{df['volume'].iloc[-1]/1e6:.1f}M"
        st.markdown(metric_card("VOLUME", vol, help_text="Shares traded", icon="📦"), unsafe_allow_html=True)
    with r4:
        atr_v = df_feat["atr"].iloc[-1] if "atr" in df_feat else 0
        st.markdown(metric_card("ATR", f"{atr_v:.2f}", help_text="Volatility index", icon="⚡"), unsafe_allow_html=True)

    # ─── Prediction Card ───
    if ens_dir is not None:
        st.markdown(pred_card(ens_dir, conf, details), unsafe_allow_html=True)
    else:
        st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;">
            <div class="metric-label">ENSEMBLE SIGNAL</div>
            <div style="margin-top:1rem;"><span class="tag tag-neutral">Train model to see prediction</span></div>
        </div>""", unsafe_allow_html=True)

    # ─── Chart ───
    st.markdown('<div class="section-header">Price Action</div>', unsafe_allow_html=True)
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        row_heights=[0.65, 0.15, 0.20])
    lookback = min(90, len(df))
    fig.add_trace(go.Candlestick(x=df.index[-lookback:], open=df["open"][-lookback:],
                 high=df["high"][-lookback:], low=df["low"][-lookback:],
                 close=df["close"][-lookback:],
                 increasing_line_color="#10b981", decreasing_line_color="#f43f5e",
                 increasing_fillcolor="rgba(16,185,129,0.3)", decreasing_fillcolor="rgba(244,63,94,0.3)",
                 name="Price"), row=1, col=1)
    if "sma_20" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["sma_20"][-lookback:],
                     name="SMA 20", line=dict(color="#8b5cf6", width=1.5, dash="dot")), row=1, col=1)
    if "sma_50" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["sma_50"][-lookback:],
                     name="SMA 50", line=dict(color="#f59e0b", width=1.5, dash="dot")), row=1, col=1)
    if "rsi" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["rsi"][-lookback:],
                     name="RSI", line=dict(color="#22d3ee", width=1.5)), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(244,63,94,0.3)", line_width=1, row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(16,185,129,0.3)", line_width=1, row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.add_trace(go.Bar(x=df.index[-lookback:], y=df["volume"][-lookback:],
                 name="Volume", marker_color="rgba(34,211,238,0.2)", marker_line_width=0), row=3, col=1)
    fig.update_layout(
        height=500, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(241,245,249,0.7)", family="Inter, sans-serif", size=11),
        margin=dict(l=0, r=0, t=20, b=0),
        hovermode="x unified",
        hoverlabel=dict(bgcolor="rgba(17,24,39,0.95)", font_size=12, font_family="JetBrains Mono, monospace"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(51,65,85,0.3)", zerolinecolor="rgba(51,65,85,0.3)"),
        yaxis=dict(gridcolor="rgba(51,65,85,0.3)", zerolinecolor="rgba(51,65,85,0.3)"),
    )
    st.plotly_chart(fig, width='stretch')

    # ─── Bottom Section ───
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        if "rsi" in df_feat:
            rsi_v = df_feat["rsi"].iloc[-1]
            rsi_s = "Overbought" if rsi_v > 70 else "Oversold" if rsi_v < 30 else "Neutral"
            rsi_c = "#f43f5e" if rsi_v > 70 else "#10b981" if rsi_v < 30 else "#94a3b8"
            cols[0].markdown(f'<div class="stat-item"><div class="metric-label">RSI (14)</div><div class="metric-val" style="color:{rsi_c}">{rsi_v:.1f}</div><div style="font-size:0.7rem;color:{rsi_c};font-weight:600;">{rsi_s}</div></div>', unsafe_allow_html=True)
        if "macd" in df_feat:
            macd_v = df_feat["macd"].iloc[-1]
            macd_sig = df_feat["macd_signal"].iloc[-1]
            macd_s = "Bullish" if macd_v > macd_sig else "Bearish"
            macd_c = "#10b981" if macd_v > macd_sig else "#f43f5e"
            cols[1].markdown(f'<div class="stat-item"><div class="metric-label">MACD</div><div class="metric-val" style="color:{macd_c}">{macd_v:.2f}</div><div style="font-size:0.7rem;color:{macd_c};font-weight:600;">{macd_s}</div></div>', unsafe_allow_html=True)
        if "bb_width" in df_feat:
            bb = df_feat["bb_width"].iloc[-1]
            cols[2].markdown(f'<div class="stat-item"><div class="metric-label">BB Width</div><div class="metric-val">{bb:.3f}</div></div>', unsafe_allow_html=True)

        if models_exist(ticker):
            try:
                _, xgb_m, _, _, _, _, _ = load_models(ticker)
                if hasattr(xgb_m, "feature_importances_"):
                    imp = xgb_m.feature_importances_
                    feat_used = [c for c in FEATURE_COLS if c in df_feat.columns]
                    fi_df = pd.DataFrame({"f": feat_used, "i": imp}).sort_values("i", ascending=False).head(8)
                    fig_fi = go.Figure(go.Bar(x=fi_df["i"][::-1], y=fi_df["f"][::-1],
                        orientation="h", marker=dict(color=["#22d3ee" if v > fi_df["i"].mean() else "rgba(148,163,184,0.15)" for v in fi_df["i"][::-1]])))
                    fig_fi.update_layout(template="plotly_dark", height=260,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0), xaxis_visible=False, yaxis_title=None,
                        yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
                    st.markdown('<div class="section-header" style="margin-top:1rem">Feature Importance</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig_fi, width='stretch')
            except Exception:
                pass

    with c_right:
        st.markdown('<div class="section-header">Recent Price Data</div>', unsafe_allow_html=True)
        display_df = df.tail(7)[["open","high","low","close","volume"]].copy()
        display_df.index = display_df.index.strftime("%d %b")
        display_df.columns = [c.capitalize() for c in display_df.columns]
        display_df["Chg%"] = display_df["Close"].pct_change().mul(100).round(1).fillna(0)
        display_df["Close"] = display_df["Close"].round(2)
        display_df["Volume"] = (display_df["Volume"] / 1e6).round(1)
        st.dataframe(display_df.iloc[::-1], width='stretch')

        if ens_dir is not None and details:
            st.markdown('<div class="section-header" style="margin-top:1rem">Model Votes</div>', unsafe_allow_html=True)
            mc = st.columns(5)
            model_names = [("LSTM","lstm_dir"),("GRU","gru_dir"),("Transformer","transformer_dir"),("XGBoost","xgb_dir"),("LightGBM","lgb_dir")]
            for i, (nm, k) in enumerate(model_names):
                if k in details:
                    d = details[k]
                    c = "#10b981" if d == 1 else "#f43f5e"
                    a = "▲ UP" if d == 1 else "▼ DN"
                    mc[i].markdown(f'<div class="stat-item" style="padding:0.6rem 0.4rem;"><div class="metric-label">{nm}</div><div style="font-weight:700;color:{c};font-size:0.8rem;font-family:var(--font-mono);">{a}</div></div>', unsafe_allow_html=True)

    # Long term chart
    st.markdown('<div class="section-header" style="margin-top:0.5rem">Long Term Trend</div>', unsafe_allow_html=True)
    sma_200 = df["close"].rolling(200).mean()
    # ─── Long Term Trend ───
    st.markdown('<div class="section-header" style="margin-top:0.5rem">Long Term Trend</div>', unsafe_allow_html=True)
    sma_200 = df["close"].rolling(200).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="rgba(241,245,249,0.9)", width=1.5)))
    fig2.add_trace(go.Scatter(x=df.index, y=df_feat["sma_50"], name="SMA 50", line=dict(color="#8b5cf6", width=1.5, dash="dot")))
    fig2.add_trace(go.Scatter(x=df.index, y=sma_200, name="SMA 200", line=dict(color="#f59e0b", width=1.5, dash="dot")))
    fig2.update_layout(template="plotly_dark", height=280,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(family="Inter, sans-serif"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=10)),
        xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
    st.plotly_chart(fig2, width='stretch')


def tab_portfolio():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Portfolio</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Performance Overview</span>
    </div>""", unsafe_allow_html=True)
    p = st.session_state.portfolio
    stats = p.get_stats()
    if not stats:
        st.markdown(f"""<div class="glass" style="text-align:center;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">📊</div>
            <div style="font-size:1rem;font-weight:600;color:var(--text-primary);margin-bottom:0.3rem;">No trades yet</div>
            <div style="font-size:0.85rem;color:var(--text-muted);">Go to <b>Backtest</b> to run a simulation</div>
        </div>""", unsafe_allow_html=True)
        return

    r1, r2, r3, r4 = st.columns(4)
    r1.markdown(metric_card("TOTAL RETURN", f'{stats["total_return"]:.1%}', icon="📈"), unsafe_allow_html=True)
    r2.markdown(metric_card("SHARPE RATIO", f'{stats["sharpe_ratio"]:.2f}', icon="⚡"), unsafe_allow_html=True)
    r3.markdown(metric_card("MAX DRAWDOWN", f'{stats["max_drawdown"]:.1f}%', icon="📉"), unsafe_allow_html=True)
    r4.markdown(metric_card("WIN RATE", f'{stats["win_rate"]:.1f}%', icon="🎯"), unsafe_allow_html=True)

    eq = pd.DataFrame(p.equity_curve)
    if len(eq) > 1:
        eq["date"] = pd.to_datetime(eq["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
            fill="tozeroy", line=dict(color="#10b981", width=2),
            fillcolor="rgba(16,185,129,0.08)"))
        fig.update_layout(template="plotly_dark", height=320,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            margin=dict(l=0,r=0,t=10,b=0), yaxis_title="Value (₹)",
            xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
        st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    c1.markdown(metric_card("CASH", f'₹{stats["cash_remaining"]:,.2f}', icon="💰"), unsafe_allow_html=True)
    c2.markdown(metric_card("HOLDINGS VALUE", f'₹{stats["holdings_value"]:,.2f}', icon="📦"), unsafe_allow_html=True)

    if p.trades:
        st.markdown(f'<div class="section-header" style="margin-top:1rem">Trade Log ({len(p.trades)})</div>', unsafe_allow_html=True)
        df_t = pd.DataFrame(p.trades).iloc[::-1]
        if "pnl" in df_t.columns:
            df_t["pnl"] = df_t["pnl"].round(2)
        st.dataframe(df_t, width='stretch', hide_index=True)


def tab_backtest():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Walk-Forward Backtest</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:var(--text-muted);margin-top:-1rem;margin-bottom:1.5rem;">Out-of-sample testing: trains on 3 years, tests on 1 year, rolls forward — zero data leakage</div>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock", NSE_STOCKS, key="bt_ticker", label_visibility="collapsed")
    c1, c2, c3 = st.columns(3)
    with c1:
        capital = st.number_input("Capital (₹)", 10000, 10_000_000, 100000, step=50000)
    with c2:
        stop_loss = st.slider("Stop Loss %", 1, 20, 5)
    with c3:
        take_profit = st.slider("Take Profit %", 5, 50, 15)

    if st.button("▶ Run Walk-Forward Backtest", type="primary", width='stretch'):
        with st.spinner(f"Training & testing {ticker} across rolling windows... (2-5 min)"):
            df = fetch_stock_data(ticker, period="5y")
            df_feat = add_technical_indicators(df, ticker=ticker)
            from src.trainer import FEATURE_COLS
            metrics, portfolio, results = run_walk_forward_backtest(
                ticker, df_feat, FEATURE_COLS,
                train_years=3, test_years=1, step_months=6,
                initial_capital=capital,
                stop_loss_pct=stop_loss / 100,
                take_profit_pct=take_profit / 100,
            )
            st.session_state.portfolio = portfolio

        if metrics:
            st.markdown('<div class="section-header" style="margin-top:1.5rem;">Walk-Forward Results (Out-of-Sample)</div>', unsafe_allow_html=True)
            r1, r2, r3, r4, r5 = st.columns(5)
            acc = metrics.get("ensemble_accuracy", 0)
            acc_c = "#10b981" if acc > 0.5 else "#f43f5e"
            r1.markdown(f'<div class="stat-item"><div class="metric-label">ENSEMBLE ACC</div><div class="metric-val" style="font-size:1.3rem;color:{acc_c}">{acc:.1%}</div></div>', unsafe_allow_html=True)

            sim_ret = metrics.get("simulated_annual_return", 0)
            ret_c = "#10b981" if sim_ret > 0 else "#f43f5e"
            r2.markdown(f'<div class="stat-item"><div class="metric-label">SIM. ANNUAL</div><div class="metric-val" style="font-size:1.3rem;color:{ret_c}">{sim_ret:.1%}</div></div>', unsafe_allow_html=True)

            r3.markdown(f'<div class="stat-item"><div class="metric-label">SIM. SHARPE</div><div class="metric-val" style="font-size:1.3rem;">{metrics.get("simulated_sharpe", 0):.2f}</div></div>', unsafe_allow_html=True)

            r4.markdown(f'<div class="stat-item"><div class="metric-label">TEST DAYS</div><div class="metric-val" style="font-size:1.3rem;">{metrics.get("total_test_days", 0)}</div></div>', unsafe_allow_html=True)

            r5.markdown(f'<div class="stat-item"><div class="metric-label">WINDOWS</div><div class="metric-val" style="font-size:1.3rem;">{metrics.get("n_windows", 0)}</div></div>', unsafe_allow_html=True)

            if results:
                st.markdown('<div class="section-header" style="margin-top:1.5rem;">Prediction Distribution</div>', unsafe_allow_html=True)
                rdf = pd.DataFrame(results)
                fig = go.Figure()
                buys = rdf[rdf["predicted"] == 1]
                sells = rdf[rdf["predicted"] == 0]
                fig.add_trace(go.Histogram(x=buys["ensemble_prob"], name="BUY", marker_color="#10b981", opacity=0.7))
                fig.add_trace(go.Histogram(x=sells["ensemble_prob"], name="SELL", marker_color="#f43f5e", opacity=0.7))
                fig.update_layout(template="plotly_dark", height=280, barmode="overlay",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif"),
                    margin=dict(l=0,r=0,t=10,b=0), xaxis_title="Ensemble Probability",
                    xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
                st.plotly_chart(fig, width='stretch')

                acc_by_model = {}
                for m in ["lstm", "gru", "transformer"]:
                    correct = (rdf[m] == rdf["actual"]).sum()
                    acc_by_model[m.upper()] = correct / len(rdf) * 100
                xgb_correct = ((rdf["xgb_prob"] > 0.5).astype(int) == rdf["actual"]).sum()
                acc_by_model["XGBoost"] = xgb_correct / len(rdf) * 100

                st.markdown('<div class="section-header" style="margin-top:1.5rem;">Individual Model Accuracy</div>', unsafe_allow_html=True)
                mc = st.columns(4)
                for i, (nm, a) in enumerate(acc_by_model.items()):
                    c = "#10b981" if a > 50 else "#f43f5e"
                    mc[i].markdown(f'<div class="stat-item" style="padding:0.6rem;"><div class="metric-label">{nm}</div><div class="metric-val" style="font-size:1.1rem;color:{c}">{a:.1f}%</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="section-header" style="margin-top:1.5rem;">Per-Window Breakdown</div>', unsafe_allow_html=True)
                window_size = len(results) // metrics.get("n_windows", 1)
                for w in range(metrics.get("n_windows", 0)):
                    start = w * window_size
                    end = min(start + window_size, len(results))
                    window_results = results[start:end]
                    if window_results:
                        wr = sum(1 for r in window_results if r["predicted"] == r["actual"]) / len(window_results) * 100
                        first_date = window_results[0]["date"]
                        last_date = window_results[-1]["date"]
                        bar_c = "#10b981" if wr > 50 else "#f43f5e"
                        st.markdown(f'<div style="display:flex;align-items:center;gap:1rem;margin:0.3rem 0;"><span style="color:var(--text-muted);min-width:200px;font-size:0.85rem;font-family:var(--font-mono);">{first_date} → {last_date}</span><div style="flex:1;height:6px;background:var(--bg-secondary);border-radius:3px;overflow:hidden;"><div style="width:{wr}%;height:100%;background:{bar_c};border-radius:3px;"></div></div><span style="font-weight:600;min-width:50px;text-align:right;font-family:var(--font-mono);font-size:0.85rem;">{wr:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚠️</div>
                <div style="font-weight:600;color:var(--text-primary);">Insufficient Data</div>
                <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">Not enough data for walk-forward backtest. Try a longer period.</div>
            </div>""", unsafe_allow_html=True)


def tab_scanner():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Scanner</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Multi-Stock Analysis</span>
    </div>""", unsafe_allow_html=True)
    trained = [t for t in NSE_STOCKS if models_exist(t)]
    st.markdown(f'<div style="font-size:0.85rem;color:var(--text-muted);margin-bottom:1rem;">Scanning <span style="color:var(--accent-cyan);font-weight:600;">{len(trained)}</span> of {len(NSE_STOCKS)} NSE stocks</div>', unsafe_allow_html=True)
    if not trained:
        st.markdown(f"""<div class="glass" style="text-align:center;padding:3rem;">
            <div style="font-size:2rem;margin-bottom:0.5rem;">🔍</div>
            <div style="font-weight:600;color:var(--text-primary);">No Trained Models</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">Train models in the <b>Predictions</b> tab first</div>
        </div>""", unsafe_allow_html=True)
        return

    if st.button("🔍 Scan Now", type="primary", width='stretch'):
        results = []
        prog = st.progress(0)
        for i, ticker in enumerate(trained):
            try:
                df = fetch_stock_data(ticker, period="6mo")
                df_feat = add_technical_indicators(df, ticker=ticker)
                lstm, gru, transformer, xgb, scaler, feat, lgb = load_models(ticker)
                d, conf, det = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat, lgb_model=lgb)
                price = df["close"].iloc[-1]
                chg5 = (df["close"].iloc[-1]/df["close"].iloc[-5]-1)*100
                rsi = df_feat["rsi"].iloc[-1] if "rsi" in df_feat else 50
                results.append({"Ticker": ticker.replace(".NS",""), "Signal": "BUY ▲" if d==1 else "SELL ▼",
                    "Conf": f"{conf:.0f}%", "Price": f"₹{price:,.0f}", "5d": f"{chg5:+.1f}%", "RSI": f"{rsi:.0f}"})
            except Exception:
                results.append({"Ticker": ticker.replace(".NS",""), "Signal": "—", "Conf": "—", "Price": "—", "5d": "—", "RSI": "—"})
            prog.progress((i+1)/len(trained))

        df_r = pd.DataFrame(results)
        buys = df_r[df_r["Signal"].str.contains("BUY")]
        sells = df_r[df_r["Signal"].str.contains("SELL")]
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="stat-item"><div class="metric-label">BUY SIGNALS</div><div class="metric-val" style="color:#10b981">{len(buys)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="stat-item"><div class="metric-label">SELL SIGNALS</div><div class="metric-val" style="color:#f43f5e">{len(sells)}</div></div>', unsafe_allow_html=True)
        st.dataframe(df_r, width='stretch', hide_index=True)


def tab_sentiment():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">News Sentiment</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">FinBERT + Google News</span>
    </div>""", unsafe_allow_html=True)
    ticker = st.selectbox("Stock", NSE_STOCKS, key="sent_ticker")
    td = ticker.replace(".NS", "")
    if st.button(f"Analyze {td} News", type="primary", width='stretch'):
        with st.spinner(f"Analyzing news for {td}..."):
            try:
                result = get_stock_sentiment(ticker)
                score = result.get("score", 0.0)
            except Exception as e:
                st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;border-color:rgba(244,63,94,0.3);">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">❌</div>
                    <div style="font-weight:600;color:#f43f5e;">Sentiment Analysis Failed</div>
                    <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">{e}</div>
                </div>""", unsafe_allow_html=True)
                return
        lbl = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        clr = "#10b981" if score > 0.05 else "#f43f5e" if score < -0.05 else "var(--text-secondary)"
        tag_cls = "tag-buy" if score > 0.05 else "tag-sell" if score < -0.05 else "tag-neutral"
        st.markdown(f"""<div class="glass" style="text-align:center;padding:2.5rem;">
            <div class="metric-label">OVERALL SENTIMENT</div>
            <div class="metric-val" style="font-size:3rem;color:{clr};margin:0.75rem 0;">{score:+.3f}</div>
            <span class="tag {tag_cls}" style="font-size:0.85rem;padding:0.4rem 1.2rem;">{lbl}</span>
        </div>""", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.75rem;color:var(--text-muted);">Based on latest headlines • Range: -1 (bearish) to +1 (bullish)</div>', unsafe_allow_html=True)


def tab_market_pulse():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Market Pulse</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Institutional Flow + Options + MTF</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:var(--text-muted);margin-top:-1rem;margin-bottom:1.5rem;">Institutional flows, options PCR, and multi-timeframe analysis</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="section-header">FII / DII Flow (Today)</div>', unsafe_allow_html=True)
        try:
            from src.flow import fetch_fii_dii
            df_fii = fetch_fii_dii()
            if len(df_fii) > 0:
                row = df_fii.iloc[0]
                fii_net = row["fii_net"]
                dii_net = row["dii_net"]
                flow_sentiment = get_flow_sentiment(fii_net, dii_net)
                fii_c = "#10b981" if fii_net >= 0 else "#f43f5e"
                dii_c = "#10b981" if dii_net >= 0 else "#f43f5e"
                fii_sign = "+" if fii_net >= 0 else ""
                dii_sign = "+" if dii_net >= 0 else ""
                flow_tag = "tag-buy" if "Bull" in flow_sentiment else "tag-sell" if "Bear" in flow_sentiment else "tag-neutral"
                st.markdown(f'''<div class="glass">
                    <div style="display:flex;gap:2rem;justify-content:center;">
                        <div style="text-align:center">
                            <div class="metric-label">FII</div>
                            <div class="metric-val" style="font-size:1.5rem;color:{fii_c}">{fii_sign}₹{abs(fii_net):,.0f} Cr</div>
                            <div class="metric-sub" style="font-size:0.7rem;">Buy ₹{row["fii_buy"]:,.0f} / Sell ₹{row["fii_sell"]:,.0f}</div>
                        </div>
                        <div style="width:1px;background:var(--border-primary);"></div>
                        <div style="text-align:center">
                            <div class="metric-label">DII</div>
                            <div class="metric-val" style="font-size:1.5rem;color:{dii_c}">{dii_sign}₹{abs(dii_net):,.0f} Cr</div>
                            <div class="metric-sub" style="font-size:0.7rem;">Buy ₹{row["dii_buy"]:,.0f} / Sell ₹{row["dii_sell"]:,.0f}</div>
                        </div>
                    </div>
                    <div style="text-align:center;margin-top:1rem;"><span class="tag {flow_tag}">{flow_sentiment}</span></div>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass" style="text-align:center;padding:2rem;"><div class="metric-label">FII / DII DATA</div><div style="margin-top:0.75rem;color:var(--text-muted);">No data available</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="glass" style="text-align:center;padding:2rem;"><div class="metric-label">FII / DII DATA</div><div style="margin-top:0.75rem;color:var(--text-muted);">Unavailable: {e}</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Options PCR & Max Pain</div>', unsafe_allow_html=True)
        pcr = fetch_options_pcr()
        if pcr.get("pcr_oi", 0) > 0:
            pcr_v = pcr["pcr_oi"]
            pcr_c = "#10b981" if pcr_v > 1 else "#f43f5e" if pcr_v < 0.8 else "var(--text-secondary)"
            pcr_lbl = "Bullish" if pcr_v > 1 else "Bearish" if pcr_v < 0.8 else "Neutral"
            pcr_tag = "tag-buy" if pcr_v > 1 else "tag-sell" if pcr_v < 0.8 else "tag-neutral"
            st.markdown(f'''<div class="glass">
                <div style="text-align:center;">
                    <div class="metric-label">PUT-CALL RATIO ({pcr.get('symbol','NIFTY')})</div>
                    <div class="metric-val" style="font-size:2.2rem;color:{pcr_c};margin:0.5rem 0;">{pcr_v:.3f}</div>
                    <span class="tag {pcr_tag}" style="font-size:0.8rem;padding:0.3rem 1rem;">{pcr_lbl}</span>
                </div>
                <div style="display:flex;gap:2rem;justify-content:center;margin-top:1.2rem;padding-top:1rem;border-top:1px solid var(--border-primary);">
                    <div style="text-align:center"><div class="metric-label">CALL OI</div><div class="metric-sub">{pcr["call_oi"]/1e6:.1f}M</div></div>
                    <div style="width:1px;background:var(--border-primary);"></div>
                    <div style="text-align:center"><div class="metric-label">PUT OI</div><div class="metric-sub">{pcr["put_oi"]/1e6:.1f}M</div></div>
                    <div style="width:1px;background:var(--border-primary);"></div>
                    <div style="text-align:center"><div class="metric-label">MAX PAIN</div><div class="metric-sub">₹{pcr['max_pain']:,.0f}</div></div>
                </div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass" style="text-align:center;padding:2rem;"><div class="metric-label">OPTIONS DATA</div><div style="margin-top:0.75rem;color:var(--text-muted);">Market closed — PCR only available during trading hours</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1.5rem">Multi-Timeframe Analysis</div>', unsafe_allow_html=True)
    tf_ticker = st.selectbox("Stock", NSE_STOCKS, key="tf_ticker", label_visibility="collapsed")
    if st.button("Analyze Timeframes", type="primary", key="tf_btn", width='stretch'):
        with st.spinner(f"Analyzing {tf_ticker} across 4 timeframes..."):
            mtf = fetch_mtf_data(tf_ticker)
            signal = get_combined_signal(mtf)

        if signal["timeframes"]:
            sig_c = "#10b981" if signal["signal"] == "Bullish" else "#f43f5e" if signal["signal"] == "Bearish" else "var(--text-secondary)"
            sig_tag = "tag-buy" if signal['signal']=='Bullish' else "tag-sell" if signal['signal']=='Bearish' else "tag-neutral"
            st.markdown(f'''<div class="glass" style="text-align:center;padding:2rem;">
                <div class="metric-label">COMBINED SIGNAL ({tf_ticker.replace('.NS','')})</div>
                <div class="metric-val" style="font-size:2.2rem;color:{sig_c};margin:0.75rem 0;">{signal['signal']}</div>
                <span class="tag {sig_tag}" style="font-size:0.8rem;">Confidence: {signal['confidence']:.0f}%</span>
            </div>''', unsafe_allow_html=True)

            tf_cols = st.columns(4)
            tf_names = {"15m": "15 MIN", "1h": "1 HOUR", "daily": "DAILY", "weekly": "WEEKLY"}
            for i, (tf, name) in enumerate(tf_names.items()):
                if tf in signal["timeframes"]:
                    sig = signal["timeframes"][tf]
                    tc = "#10b981" if sig["signal"] == "Bullish" else "#f43f5e" if sig["signal"] == "Bearish" else "var(--text-secondary)"
                    ind_str = "<br>".join([f'<span style="font-size:0.7rem;">{k}: {v}</span>' for k, v in sig["indicators"].items()])
                    tf_cols[i].markdown(f'''<div class="stat-item" style="text-align:center;padding:1rem;">
                        <div class="metric-label">{name}</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{tc};margin:0.5rem 0;">{sig['signal']}</div>
                        <div class="metric-sub" style="font-size:0.7rem;">Strength: {sig['strength']:.0%}</div>
                        <div style="margin-top:0.75rem;">{ind_str}</div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;">
                <div style="font-size:1.5rem;margin-bottom:0.5rem;">📊</div>
                <div style="font-weight:600;color:var(--text-primary);">No Timeframe Data</div>
                <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">Click "Analyze Timeframes" to see multi-timeframe signals</div>
            </div>""", unsafe_allow_html=True)


def tab_optimizer():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Portfolio Optimizer</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">MVO + Black-Litterman</span>
    </div>""", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.8rem;color:var(--text-muted);margin-top:-1rem;margin-bottom:1.5rem;">Mean-Variance Optimization + Black-Litterman — finds optimal allocation</div>', unsafe_allow_html=True)

    selected = st.multiselect("Select Stocks", NSE_STOCKS, default=NSE_STOCKS[:3], key="opt_stocks", label_visibility="collapsed")
    if len(selected) < 2:
        st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;">
            <div style="font-size:1.5rem;margin-bottom:0.5rem;">📊</div>
            <div style="font-weight:600;color:var(--text-primary);">Select at least 2 stocks</div>
            <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">Choose stocks to optimize portfolio allocation</div>
        </div>""", unsafe_allow_html=True)
        return

    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2, key="opt_period")

    if st.button("⚡ Optimize Portfolio", type="primary", width='stretch', key="opt_btn"):
        with st.spinner("Fetching prices and optimizing..."):
            prices = pd.DataFrame()
            for t in selected:
                df = fetch_stock_data(t, period=period)
                prices[t] = df["close"]
            prices = prices.dropna()

            if len(prices) < 60:
                st.markdown(f"""<div class="glass" style="text-align:center;padding:2rem;">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem;">⚠️</div>
                    <div style="font-weight:600;color:var(--text-primary);">Insufficient Price History</div>
                    <div style="font-size:0.85rem;color:var(--text-muted);margin-top:0.3rem;">Need at least 60 days of data. Try a longer period.</div>
                </div>""", unsafe_allow_html=True)
                return

            result = optimize_portfolio(prices)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Optimal Portfolios</div>', unsafe_allow_html=True)
        for ptype, label in [("max_sharpe", "MAX SHARPE"), ("min_variance", "MIN VARIANCE"), ("black_litterman", "BLACK-LITTERMAN")]:
            p = result[ptype]
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f'<span style="font-weight:700;font-size:0.85rem;letter-spacing:0.03em;">{label}</span>', unsafe_allow_html=True)
                for i, t in enumerate(selected):
                    w = p["weights"][i]
                    bar = "█" * int(w * 30)
                    st.markdown(f"  {t.replace('.NS','')}: **{w:.1%}** `{bar}`")
            with c2:
                st.markdown(f'<div class="stat-item"><div class="metric-label">EXP. RETURN</div><div class="metric-val" style="font-size:1.2rem;">{p["return"]:.1%}</div></div>', unsafe_allow_html=True)
            with c3:
                st.markdown(f'<div class="stat-item"><div class="metric-label">VOLATILITY</div><div class="metric-val" style="font-size:1.2rem;">{p["volatility"]:.1%}</div></div>', unsafe_allow_html=True)
            st.markdown('<hr style="border:none;border-top:1px solid var(--border-primary);margin:1rem 0;">', unsafe_allow_html=True)

        if result["efficient_frontier"]:
            ef = pd.DataFrame(result["efficient_frontier"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ef["volatility"], y=ef["return"], mode="lines+markers",
                name="Efficient Frontier", line=dict(color="#22d3ee", width=2),
                marker=dict(size=6, color="#22d3ee")))
            ms = result["max_sharpe"]
            fig.add_trace(go.Scatter(x=[ms["volatility"]], y=[ms["return"]], mode="markers",
                name="Max Sharpe", marker=dict(color="#8b5cf6", size=16, symbol="star",
                line=dict(width=2, color="white"))))
            fig.update_layout(template="plotly_dark", height=380,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"),
                xaxis_title="Volatility", yaxis_title="Return",
                margin=dict(l=0,r=0,t=20,b=0),
                xaxis=dict(gridcolor="rgba(51,65,85,0.3)"), yaxis=dict(gridcolor="rgba(51,65,85,0.3)"))
            st.plotly_chart(fig, width='stretch')


def tab_holdings():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Holdings Tracker</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Zerodha Portfolio Analysis</span>
    </div>""", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Zerodha Holdings CSV", type=["csv"], key="holdings_upload")

    if uploaded:
        holdings_df = parse_holdings_csv(uploaded)
        stats = compute_portfolio_stats(holdings_df)

        if stats:
            r1, r2, r3, r4 = st.columns(4)
            pnl_c = "#10b981" if stats["total_pnl"] >= 0 else "#f43f5e"
            r1.markdown(metric_card("INVESTED", f'₹{stats["total_invested"]:,.0f}', icon="💰"), unsafe_allow_html=True)
            r2.markdown(metric_card("CURRENT VALUE", f'₹{stats["total_current"]:,.0f}', icon="📊"), unsafe_allow_html=True)
            r3.markdown(f'<div class="stat-item"><div class="metric-label">P&L</div><div class="metric-val" style="color:{pnl_c}">₹{stats["total_pnl"]:+,.0f}</div></div>', unsafe_allow_html=True)
            r4.markdown(f'<div class="stat-item"><div class="metric-label">RETURN</div><div class="metric-val" style="color:{pnl_c}">{stats["total_return_pct"]:+.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header" style="margin-top:1rem;">Category Allocation</div>', unsafe_allow_html=True)
            cat_data = []
            for cat, info in sorted(stats["categories"].items(), key=lambda x: -x[1]["weight"]):
                cat_data.append({"Category": cat, "Value": f"₹{info['value']:,.0f}", "Weight": f"{info['weight']:.1%}", "Return": f"{info['return_pct']:+.1f}%"})
            st.dataframe(pd.DataFrame(cat_data), width='stretch', hide_index=True)

            if stats["categories"]:
                cat_labels = list(stats["categories"].keys())
                cat_values = [stats["categories"][c]["value"] for c in cat_labels]
                fig = go.Figure(go.Pie(labels=cat_labels, values=cat_values,
                    hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)))
                fig.update_layout(template="plotly_dark", height=380,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(family="Inter, sans-serif"),
                    margin=dict(l=0,r=0,t=10,b=0), showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, font=dict(size=10)))
                st.plotly_chart(fig, width='stretch')

            st.markdown('<div class="section-header">Holdings Detail</div>', unsafe_allow_html=True)
            h_data = []
            for h in stats["holdings"]:
                h_data.append({
                    "Fund": h["name"][:35],
                    "Invested": f"₹{h['invested']:,.0f}",
                    "Current": f"₹{h['current_value']:,.0f}",
                    "P&L": f"₹{h['pnl']:+,.0f}",
                    "Return": f"{h['return_pct']:+.1f}%",
                    "Weight": f"{h['weight']:.1%}",
                })
            st.dataframe(pd.DataFrame(h_data), width='stretch', hide_index=True)

            st.markdown('<div class="section-header">Risk Assessment</div>', unsafe_allow_html=True)
            returns_list = [h["return_pct"] for h in stats["holdings"]]
            weights_list = [h["weight"] for h in stats["holdings"]]
            herfindahl = sum(w**2 for w in weights_list)
            top3_weight = sum(sorted(weights_list, reverse=True)[:3])
            risk_notes = []
            if herfindahl > 0.25:
                risk_notes.append(("High concentration", "Herfindahl index > 0.25 — portfolio is concentrated", "tag-sell"))
            if top3_weight > 0.60:
                risk_notes.append(("Top-heavy", f"Top 3 holdings = {top3_weight:.0%} of portfolio", "tag-warn"))
            if any(h["return_pct"] < -5 for h in stats["holdings"]):
                risk_notes.append(("Underperformers", "Some holdings with >5% loss", "tag-sell"))
            if not risk_notes:
                risk_notes.append(("Healthy", "Portfolio risk profile looks reasonable", "tag-buy"))
            for title, note, tag in risk_notes:
                st.markdown(f'<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:0.5rem;"><span class="tag {tag}">{title}</span><span style="font-size:0.85rem;color:var(--text-secondary);">{note}</span></div>', unsafe_allow_html=True)


def tab_risk():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Risk Dashboard</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">VaR + Kelly + Portfolio Risk</span>
    </div>""", unsafe_allow_html=True)

    ticker = st.selectbox("Stock", NSE_STOCKS, key="risk_ticker", label_visibility="collapsed")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="risk_period")

    if st.button("⚡ Analyze Risk", type="primary", width='stretch', key="risk_btn"):
        with st.spinner("Computing risk metrics..."):
            df = fetch_stock_data(ticker, period=period)
            returns = df["close"].pct_change().dropna().values
            equity = (1 + returns).cumprod() * 100000
            report = generate_risk_report(returns, equity)
            regime = detect_regime(df["close"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Market Regime</div>', unsafe_allow_html=True)
            regime_c = "#10b981" if regime["regime"] == "Bull" else "#f43f5e" if regime["regime"] == "Bear" else "#f59e0b"
            regime_tag = "regime-bull" if regime["regime"]=="Bull" else "regime-bear" if regime["regime"]=="Bear" else "regime-neutral"
            rec = regime["recommendation"]
            st.markdown(f'''<div class="glass" style="text-align:center;padding:2rem;">
                <div class="metric-label">CURRENT REGIME</div>
                <div class="regime-badge {regime_tag}" style="font-size:1.5rem;margin:1rem auto;">{regime["regime"]}</div>
                <div class="metric-sub" style="margin-top:0.75rem;">{regime["confidence"]:.0f}% confidence</div>
            </div>''', unsafe_allow_html=True)
            st.markdown(f'''<div class="glass">
                <div class="section-header">Recommendation</div>
                <div style="font-weight:700;font-size:1rem;margin:0.5rem 0;">{rec["action"]}</div>
                <div class="metric-sub">Allocation: {rec["allocation"]}</div>
                <div class="metric-sub">Risk Level: {rec["risk_level"]}</div>
            </div>''', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">Risk Metrics</div>', unsafe_allow_html=True)
            metrics_grid = [
                ("Annual Return", f"{report['annual_return']:.1f}%", "#10b981" if report['annual_return'] > 0 else "#f43f5e"),
                ("Annual Volatility", f"{report['annual_volatility']:.1f}%", "var(--text-secondary)"),
                ("Sharpe Ratio", f"{report['sharpe']:.3f}", "#10b981" if report['sharpe'] > 1 else "#f43f5e"),
                ("Sortino Ratio", f"{report['sortino']:.3f}", "#10b981" if report['sortino'] > 1 else "#f43f5e"),
                ("Max Drawdown", f"{report['max_drawdown']:.1f}%", "#f43f5e"),
                ("VaR (95%)", f"{report['var_95']:.2f}%", "#f43f5e"),
                ("CVaR (95%)", f"{report['cvar_95']:.2f}%", "#f43f5e"),
                ("Calmar Ratio", f"{report['calmar']:.3f}", "#10b981" if report['calmar'] > 1 else "#f59e0b"),
                ("Win Days", f"{report['positive_days_pct']:.0f}%", "#10b981" if report['positive_days_pct'] > 50 else "#f43f5e"),
                ("Best Day", f"{report['best_day']:.2f}%", "#10b981"),
                ("Worst Day", f"{report['worst_day']:.2f}%", "#f43f5e"),
                ("Skewness", f"{report['skewness']:.2f}", "var(--text-secondary)"),
            ]
            for i in range(0, len(metrics_grid), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i+j < len(metrics_grid):
                        label, val, color = metrics_grid[i+j]
                        cols[j].markdown(f'<div class="stat-item" style="padding:0.75rem;"><div class="metric-label">{label}</div><div class="metric-val" style="font-size:1.1rem;color:{color}">{val}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Drawdown Chart</div>', unsafe_allow_html=True)
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-len(drawdown):], y=drawdown, fill="tozeroy",
            line=dict(color="#f43f5e", width=1.5), fillcolor="rgba(244,63,94,0.1)", name="Drawdown"))
        fig.update_layout(template="plotly_dark", height=280,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"),
            yaxis_title="Drawdown %", margin=dict(l=0,r=0,t=10,b=0),
            xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
        st.plotly_chart(fig, width='stretch')

        st.markdown('<div class="section-header">Kelly Criterion</div>', unsafe_allow_html=True)
        win_days = (returns > 0).sum()
        lose_days = (returns < 0).sum()
        avg_win = returns[returns > 0].mean() if win_days > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if lose_days > 0 else 0.001
        win_rate = win_days / len(returns)
        kelly = kelly_criterion(win_rate, avg_win, avg_loss)
        st.markdown(f'''<div class="glass">
            <div class="metric-label">OPTIMAL POSITION SIZE (KELLY)</div>
            <div class="metric-val" style="font-size:2rem;margin:0.5rem 0;">{kelly:.1%}</div>
            <div style="display:flex;gap:1.5rem;margin-top:0.5rem;">
                <div class="metric-sub">Win rate: {win_rate:.1%}</div>
                <div class="metric-sub">Avg win: {avg_win:.3f}</div>
                <div class="metric-sub">Avg loss: {avg_loss:.3f}</div>
            </div>
        </div>''', unsafe_allow_html=True)


def tab_volatility():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Volatility Analysis</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Forecast + Regime + Position Sizing</span>
    </div>""", unsafe_allow_html=True)

    from src.volatility import full_volatility_analysis

    ticker = st.selectbox("Stock", NSE_STOCKS, key="vol_ticker", label_visibility="collapsed")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="vol_period")

    if st.button("⚡ Analyze Volatility", type="primary", width='stretch', key="vol_btn"):
        with st.spinner("Computing volatility metrics..."):
            df = fetch_stock_data(ticker, period=period)
            df_feat = add_technical_indicators(df, ticker=ticker)
            result = full_volatility_analysis(df_feat)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Current Volatility</div>', unsafe_allow_html=True)
            vol = result["current"]
            regime = result["regime"]

            regime_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#f43f5e"}[regime["current_regime"]]
            st.markdown(f'''<div class="glass" style="text-align:center;padding:1.5rem;">
                <div class="metric-label">VOLATILITY REGIME</div>
                <div style="font-size:1.8rem;font-weight:800;color:{regime_color};margin:0.5rem 0;">{regime["current_regime"]}</div>
                <div class="metric-sub">Current: {regime["current_vol"]:.1%} annualized</div>
                <div class="metric-sub">Percentile: {regime["vol_percentile"]:.0f}%</div>
            </div>''', unsafe_allow_html=True)

            vol_grid = [
                ("Historical", f"{vol['hist_vol']:.1%}"),
                ("EWMA", f"{vol['ewma_vol']:.1%}"),
                ("Parkinson", f"{vol['parkinson_vol']:.1%}"),
                ("Garman-Klass", f"{vol['garman_klass_vol']:.1%}"),
                ("Yang-Zhang", f"{vol['yang_zhang_vol']:.1%}"),
                ("ATR", f"{vol['atr_pct']:.2%}"),
                ("BB Width", f"{vol['bollinger_bandwidth']:.3f}"),
                ("BB %B", f"{vol['bollinger_pct_b']:.2f}"),
            ]
            for i in range(0, len(vol_grid), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i+j < len(vol_grid):
                        label, val = vol_grid[i+j]
                        cols[j].markdown(f'<div class="stat-item" style="padding:0.5rem;"><div class="metric-label">{label}</div><div class="metric-val">{val}</div></div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">5-Day Forecast</div>', unsafe_allow_html=True)
            forecast = result["forecast"]
            fig = go.Figure()
            days = list(range(1, forecast["horizon"] + 1))
            fig.add_trace(go.Scatter(x=days, y=forecast["forecast_vols"], mode="lines+markers",
                line=dict(color="#22d3ee", width=2), name="Forecast"))
            fig.add_hline(y=forecast["current_vol"], line_dash="dash", line_color="#f59e0b", annotation_text="Current")
            fig.add_hline(y=forecast["long_term_vol"], line_dash="dot", line_color="#10b981", annotation_text="Long-term")
            fig.update_layout(template="plotly_dark", height=250, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter, sans-serif"), yaxis_title="Annualized Vol", xaxis_title="Days",
                margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
            st.plotly_chart(fig, width='stretch')

            st.markdown('<div class="section-header">Position Sizing</div>', unsafe_allow_html=True)
            ps = result["position_sizing"]
            st.markdown(f'''<div class="glass">
                <div class="metric-label">RECOMMENDED POSITION SIZE</div>
                <div class="metric-val" style="font-size:1.8rem;margin:0.5rem 0;">{ps["position_size_pct"]:.1%}</div>
                <div class="metric-sub">Vol scalar: {ps["vol_scalar"]:.2f}x</div>
                <div class="metric-sub">{ps["reasoning"]}</div>
            </div>''', unsafe_allow_html=True)

        st.markdown('<div class="section-header" style="margin-top:1.5rem;">Volatility Cone</div>', unsafe_allow_html=True)
        cone = result["cone"]
        cone_data = []
        for w, data in cone.items():
            cone_data.append({"Window": w, "Min": data["min"], "Mean": data["mean"], "Max": data["max"], "Current": data["current"]})
        cone_df = pd.DataFrame(cone_data)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cone_df["Window"], y=cone_df["Max"], mode="lines", line=dict(color="#f43f5e", width=1), name="Max"))
        fig.add_trace(go.Scatter(x=cone_df["Window"], y=cone_df["Mean"], mode="lines", line=dict(color="#f59e0b", width=2), name="Mean"))
        fig.add_trace(go.Scatter(x=cone_df["Window"], y=cone_df["Min"], mode="lines", line=dict(color="#10b981", width=1), name="Min"))
        fig.add_trace(go.Scatter(x=cone_df["Window"], y=cone_df["Current"], mode="markers", marker=dict(color="#22d3ee", size=10), name="Current"))
        fig.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"), yaxis_title="Annualized Vol", xaxis_title="Window (days)",
            margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
        st.plotly_chart(fig, width='stretch')


def tab_ranking():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Stock Ranking</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Multi-Factor Cross-Sectional</span>
    </div>""", unsafe_allow_html=True)

    from src.ranking import rank_stocks, get_recommendation, factor_analysis

    if st.button("⚡ Rank All Stocks", type="primary", width='stretch', key="rank_btn"):
        with st.spinner("Fetching data and ranking..."):
            stock_data = {}
            for ticker in NSE_STOCKS[:10]:
                try:
                    df = fetch_stock_data(ticker, period="1y")
                    stock_data[ticker] = df
                except Exception:
                    pass

            rankings = rank_stocks(stock_data)
            fa = factor_analysis(stock_data)

        st.markdown('<div class="section-header">Stock Rankings</div>', unsafe_allow_html=True)
        for r in rankings:
            rec = get_recommendation(rankings, r["ticker"])
            rec_color = {"STRONG BUY": "#10b981", "BUY": "#22d3ee", "HOLD": "#f59e0b", "SELL": "#f97316", "STRONG SELL": "#f43f5e"}[rec["recommendation"]]
            st.markdown(f'''<div class="glass" style="padding:1rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:700;font-size:1.1rem;">#{r["rank"]} {r["ticker"]}</span>
                        <span style="color:{rec_color};font-weight:600;margin-left:1rem;">{rec["recommendation"]}</span>
                    </div>
                    <div style="text-align:right;">
                        <div class="metric-val">₹{r["current_price"]:.2f}</div>
                        <div class="metric-sub">{r["daily_return"]:+.2f}%</div>
                    </div>
                </div>
                <div style="display:flex;gap:2rem;margin-top:0.5rem;">
                    <div class="metric-sub">Momentum: {r["momentum"]["momentum_combined"]:.3f}</div>
                    <div class="metric-sub">Vol Score: {r["volatility"]["vol_score"]:.0f}</div>
                    <div class="metric-sub">Technical: {r["technical"]["technical_combined"]:.0f}</div>
                    <div class="metric-sub">Composite: {r["composite_score"]:.3f}</div>
                </div>
            </div>''', unsafe_allow_html=True)

        if not fa.get("insufficient_data"):
            st.markdown('<div class="section-header" style="margin-top:1.5rem;">Factor Analysis</div>', unsafe_allow_html=True)
            for factor, corr in fa["factor_correlations"].items():
                color = "#10b981" if corr > 0 else "#f43f5e"
                st.markdown(f'<div class="metric-sub">{factor}: {corr:.3f} {"↑" if corr > 0 else "↓"}</div>', unsafe_allow_html=True)


def tab_scenarios():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Strategy Scenarios</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Compare Different Strategies</span>
    </div>""", unsafe_allow_html=True)

    from src.scenarios import run_all_scenarios

    ticker = st.selectbox("Stock", NSE_STOCKS, key="scenario_ticker", label_visibility="collapsed")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="scenario_period")

    if st.button("⚡ Run Scenarios", type="primary", width='stretch', key="scenario_btn"):
        with st.spinner("Running all scenarios..."):
            df = fetch_stock_data(ticker, period=period)
            scenarios = run_all_scenarios(df)

        st.markdown('<div class="section-header">Scenario Results</div>', unsafe_allow_html=True)
        for s in scenarios:
            ret_color = "#10b981" if s["annualized_return"] > 0 else "#f43f5e"
            sharpe_color = "#10b981" if s["sharpe"] > 1 else "#f59e0b" if s["sharpe"] > 0 else "#f43f5e"
            st.markdown(f'''<div class="glass" style="padding:1rem;margin-bottom:0.5rem;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-weight:700;">{s["name"]}</span>
                        <span class="metric-sub" style="margin-left:1rem;">{s["n_trades"]} trades</span>
                    </div>
                    <div style="text-align:right;">
                        <span style="color:{ret_color};font-weight:700;">{s["annualized_return"]:.1%}</span>
                        <span class="metric-sub" style="margin-left:0.5rem;">Sharpe: <span style="color:{sharpe_color}">{s["sharpe"]:.2f}</span></span>
                    </div>
                </div>
                <div style="display:flex;gap:2rem;margin-top:0.5rem;">
                    <div class="metric-sub">Total Return: {s["total_return"]:.1%}</div>
                    <div class="metric-sub">Volatility: {s["annualized_vol"]:.1%}</div>
                    <div class="metric-sub">Max DD: {s["max_drawdown"]:.1%}</div>
                </div>
            </div>''', unsafe_allow_html=True)

        fig = go.Figure()
        names = [s["name"] for s in scenarios]
        sharpes = [s["sharpe"] for s in scenarios]
        colors = ["#10b981" if s > 1 else "#f59e0b" if s > 0 else "#f43f5e" for s in sharpes]
        fig.add_trace(go.Bar(x=names, y=sharpes, marker_color=colors))
        fig.update_layout(template="plotly_dark", height=350, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter, sans-serif"), yaxis_title="Sharpe Ratio", xaxis_title="Strategy",
            margin=dict(l=0,r=0,t=10,b=0), xaxis=dict(gridcolor="rgba(51,65,85,0.2)"), yaxis=dict(gridcolor="rgba(51,65,85,0.2)"))
        st.plotly_chart(fig, width='stretch')


def tab_regime_strategy():
    st.markdown(f"""<div style="display:flex;align-items:center;gap:0.75rem;margin-bottom:1.5rem;padding:1rem 1.5rem;
                background:var(--bg-card);border:1px solid var(--border-primary);border-radius:var(--radius-lg);">
        <span class="gradient-text" style="font-size:1.8rem;font-weight:800;">Regime Strategy</span>
        <span style="font-size:0.75rem;color:var(--text-muted);font-weight:500;letter-spacing:0.05em;text-transform:uppercase;">Adaptive to Market Conditions</span>
    </div>""", unsafe_allow_html=True)

    from src.regime_strategy import backtest_regime_strategy, get_regime_allocation
    from src.regime import detect_regime

    ticker = st.selectbox("Stock", NSE_STOCKS, key="regime_strat_ticker", label_visibility="collapsed")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="regime_strat_period")

    if st.button("⚡ Analyze Regime Strategy", type="primary", width='stretch', key="regime_strat_btn"):
        with st.spinner("Analyzing regime strategy..."):
            df = fetch_stock_data(ticker, period=period)
            returns = df["close"].pct_change().dropna()
            regime_result = detect_regime(returns)
            bt = backtest_regime_strategy(df, regime_result)

        c1, c2 = st.columns(2)
        with c1:
            regime = bt["regime"]
            regime_color = {"Bull": "#10b981", "Bear": "#f43f5e", "Sideways": "#f59e0b"}[regime]
            st.markdown(f'''<div class="glass" style="text-align:center;padding:2rem;">
                <div class="metric-label">CURRENT REGIME</div>
                <div style="font-size:2rem;font-weight:800;color:{regime_color};margin:0.5rem 0;">{regime}</div>
                <div class="metric-sub">Equity Allocation: {bt["equity_allocation"]:.0%}</div>
            </div>''', unsafe_allow_html=True)

            alloc = get_regime_allocation(regime, regime_result["confidence"])
            st.markdown(f'''<div class="glass">
                <div class="section-header">Strategy Recommendation</div>
                <div style="font-weight:600;margin:0.5rem 0;">{alloc["strategy"].title()}</div>
                <div class="metric-sub">{alloc["reasoning"]}</div>
            </div>''', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">Performance Comparison</div>', unsafe_allow_html=True)
            perf_grid = [
                ("Strategy Return", f"{bt['strategy_return']:.1%}", "#10b981" if bt['strategy_return'] > 0 else "#f43f5e"),
                ("Buy & Hold Return", f"{bt['buy_hold_return']:.1%}", "#10b981" if bt['buy_hold_return'] > 0 else "#f43f5e"),
                ("Strategy Annual", f"{bt['strategy_annualized']:.1%}", "var(--text-secondary)"),
                ("Buy & Hold Annual", f"{bt['buy_hold_annualized']:.1%}", "var(--text-secondary)"),
                ("Strategy Sharpe", f"{bt['strategy_sharpe']:.3f}", "#10b981" if bt['strategy_sharpe'] > 1 else "#f43f5e"),
                ("Excess Return", f"{bt['excess_return']:.1%}", "#10b981" if bt['excess_return'] > 0 else "#f43f5e"),
            ]
            for i in range(0, len(perf_grid), 2):
                cols = st.columns(2)
                for j in range(2):
                    if i+j < len(perf_grid):
                        label, val, color = perf_grid[i+j]
                        cols[j].markdown(f'<div class="stat-item" style="padding:0.5rem;"><div class="metric-label">{label}</div><div class="metric-val" style="color:{color}">{val}</div></div>', unsafe_allow_html=True)


def main():
    # (header is now rendered inside tab_predictions)
    tabs = st.tabs(["📈 Predictions", "💰 Portfolio", "🔬 Backtest", "🔍 Scanner", "📰 Sentiment", "🏛️ Market Pulse", "📊 Optimizer", "💼 Holdings", "⚡ Risk", "📉 Volatility", "🏆 Ranking", "🎯 Scenarios", "🌡️ Regime"])
    with tabs[0]: tab_predictions()
    with tabs[1]: tab_portfolio()
    with tabs[2]: tab_backtest()
    with tabs[3]: tab_scanner()
    with tabs[4]: tab_sentiment()
    with tabs[5]: tab_market_pulse()
    with tabs[6]: tab_optimizer()
    with tabs[7]: tab_holdings()
    with tabs[8]: tab_risk()
    with tabs[9]: tab_volatility()
    with tabs[10]: tab_ranking()
    with tabs[11]: tab_scenarios()
    with tabs[12]: tab_regime_strategy()
    st.markdown("""
    <div style="text-align:center;padding:2rem 0 1rem;margin-top:2rem;border-top:1px solid var(--border-primary);">
        <div style="font-size:0.7rem;color:var(--text-muted);letter-spacing:0.05em;">
            ⚠️ Educational purposes only — not financial advice. Past performance does not guarantee future results.
        </div>
        <div style="font-size:0.65rem;color:var(--text-muted);margin-top:0.5rem;opacity:0.5;">
            StoMar v1.0 — 5-Model Ensemble • Walk-Forward Backtest • Black-Litterman Optimizer
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()

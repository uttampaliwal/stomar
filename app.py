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
from src.model import load_models, models_exist, DEVICE
from src.trainer import train_for_ticker, FEATURE_COLS
from src.ensemble import predict_ensemble
from src.portfolio import Portfolio
from src.backtester import run_backtest, generate_model_signals
from src.sentiment import fetch_news_sentiment

st.set_page_config(page_title="StoMar", page_icon="⚡", layout="wide", initial_sidebar_state="collapsed")

# ─── THEME-AWARE CSS ───
st.markdown("""
<style>
/* Base theme variables - Streamlit adapts these automatically */
:root {
    --card-bg: rgba(255,255,255,0.05);
    --card-border: rgba(255,255,255,0.08);
    --card-glow: rgba(100,255,218,0.15);
    --accent: #00d4aa;
    --accent2: #7c3aed;
    --text-dim: rgba(255,255,255,0.5);
    --gradient: linear-gradient(135deg, #00d4aa 0%, #7c3aed 100%);
}

[data-theme="light"] {
    --card-bg: rgba(0,0,0,0.03);
    --card-border: rgba(0,0,0,0.08);
    --card-glow: rgba(0,212,170,0.1);
    --text-dim: rgba(0,0,0,0.4);
}

.stApp { background: var(--background-color); }

/* Glass card */
.glass {
    background: var(--card-bg);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border: 1px solid var(--card-border);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.3s ease;
}
.glass:hover { border-color: var(--accent); box-shadow: 0 0 30px var(--card-glow); }

/* Gradient text */
.gradient-text {
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-weight: 800;
}

/* Metric value */
.metric-val {
    font-size: 1.8rem; font-weight: 700;
    background: var(--gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.metric-label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: var(--text-dim); }

/* Tag */
.tag {
    display: inline-block; padding: 0.2rem 0.8rem; border-radius: 20px;
    font-size: 0.75rem; font-weight: 600; letter-spacing: 0.03em;
}
.tag-up { background: rgba(0,212,170,0.15); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.tag-down { background: rgba(255,68,68,0.15); color: #ff4444; border: 1px solid rgba(255,68,68,0.3); }
.tag-neutral { background: rgba(255,255,255,0.08); color: var(--text-dim); border: 1px solid var(--card-border); }

/* Prediction pill */
.pred-pill {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 1.2rem; border-radius: 40px;
    font-weight: 700; font-size: 1.1rem;
}
.pred-pill.up { background: rgba(0,212,170,0.12); color: #00d4aa; border: 1px solid rgba(0,212,170,0.3); }
.pred-pill.down { background: rgba(255,68,68,0.12); color: #ff4444; border: 1px solid rgba(255,68,68,0.3); }

/* Section header */
.section-header {
    font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-dim); margin-bottom: 0.5rem; padding-bottom: 0.3rem;
    border-bottom: 1px solid var(--card-border);
}

/* Buttons */
.stButton > button {
    border-radius: 40px !important; font-weight: 600 !important;
    border: 1px solid var(--card-border) !important;
    transition: all 0.3s ease !important;
}
.stButton > button:hover { border-color: var(--accent) !important; box-shadow: 0 0 20px var(--card-glow) !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] { gap: 0.5rem; }
.stTabs [data-baseweb="tab"] {
    border-radius: 40px !important; padding: 0.4rem 1rem !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: var(--gradient) !important;
    color: white !important;
}

/* Data tables */
[data-testid="stDataFrame"] { border: 1px solid var(--card-border); border-radius: 12px; overflow: hidden; }

/* Expanders */
.streamlit-expanderHeader {
    border-radius: 12px !important;
    background: var(--card-bg) !important;
    border: 1px solid var(--card-border) !important;
}

/* Metrics row */
.metrics-row { display: flex; gap: 1rem; flex-wrap: wrap; }
.metrics-row > div { flex: 1; min-width: 120px; }
</style>
""", unsafe_allow_html=True)


if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio(initial_capital=100000)
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = NSE_STOCKS[0]


@st.cache_data(ttl=3600)
def get_data(ticker, period):
    return fetch_stock_data(ticker, period=period)


def metric_card(label, value, delta=None, help_text=None):
    d = f'<span style="color:{"#00d4aa" if delta and "+" in str(delta) else "#ff4444"}">{delta}</span>' if delta else ""
    h = f'<small style="color:var(--text-dim)">{help_text}</small>' if help_text else ""
    return f'<div class="glass" style="text-align:center; padding:1rem 0.8rem;"><div class="metric-label">{label}</div><div class="metric-val">{value}</div>{d}{h}</div>'


def pred_card(direction, confidence, details=None):
    is_up = direction == 1
    cls = "up" if is_up else "down"
    arrow = "▲" if is_up else "▼"
    label = "BUY" if is_up else "SELL"
    return f"""
    <div class="glass" style="text-align:center; padding:1.2rem;">
        <div class="metric-label">Ensemble Prediction</div>
        <div class="pred-pill {cls}" style="margin:0.5rem auto;">{arrow} {label}</div>
        <div style="margin-top:0.5rem;">
            <span class="tag {'tag-up' if is_up else 'tag-down'}">{confidence:.0f}% confidence</span>
        </div>
    </div>
    """


def tab_predictions():
    col1, col2, col3, col4 = st.columns([2.5, 1, 1, 0.8])
    with col1:
        ticker = st.selectbox("Stock", NSE_STOCKS, index=NSE_STOCKS.index(st.session_state.last_ticker), label_visibility="collapsed")
        st.session_state.last_ticker = ticker
    with col2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=3, label_visibility="collapsed")
    with col3:
        refresh = st.button("↻ Refresh", width='stretch')
    with col4:
        train_btn = st.button("⚡ Train", width='stretch', type="secondary")

    market = get_market_status()
    st.caption(f"{'🟢' if market=='Open' else '🔴'} Market {market} • {ticker.replace('.NS','')} • Last updated {datetime.now().strftime('%H:%M')}")

    try:
        df = get_data(ticker, period)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        return

    df_feat = add_technical_indicators(df)
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
            lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)
            ens_dir, conf, details = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat)
        except Exception as e:
            st.warning(f"Model load failed: {e}. Retrain this stock.")

    # Top row: price + prediction
    r1, r2, r3, r4, r5 = st.columns([1.5, 1, 1, 1, 1.5])
    sign = "+" if chg >= 0 else ""
    r1.markdown(metric_card("Price", f"₹{curr:,.2f}", f"{sign}{chg:.2f} ({sign}{chg_pct:.2f}%)"), unsafe_allow_html=True)
    r2.markdown(metric_card("Range", f"₹{df['low'].iloc[-1]:,.0f}–{df['high'].iloc[-1]:,.0f}", help_text="Day high / low"), unsafe_allow_html=True)
    r3.markdown(metric_card("Volume", f"{df['volume'].iloc[-1]/1e6:.1f}M", help_text="Shares traded"), unsafe_allow_html=True)
    atr_v = df_feat["atr"].iloc[-1] if "atr" in df_feat else 0
    r4.markdown(metric_card("ATR", f"{atr_v:.2f}", help_text="Volatility"), unsafe_allow_html=True)
    if ens_dir is not None:
        r5.markdown(pred_card(ens_dir, conf, details), unsafe_allow_html=True)
    else:
        r5.markdown(f'<div class="glass" style="text-align:center;padding:1.2rem;"><div class="metric-label">Prediction</div><div style="margin-top:0.8rem;"><span class="tag tag-neutral">—</span></div></div>', unsafe_allow_html=True)

    # Chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        row_heights=[0.6, 0.18, 0.22])
    lookback = min(90, len(df))
    fig.add_trace(go.Candlestick(x=df.index[-lookback:], open=df["open"][-lookback:],
                 high=df["high"][-lookback:], low=df["low"][-lookback:],
                 close=df["close"][-lookback:],
                 increasing_line_color="#00d4aa", decreasing_line_color="#ff4444",
                 name=""), row=1, col=1)
    if "sma_20" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["sma_20"][-lookback:],
                     name="SMA20", line=dict(color="rgba(124,58,237,0.6)", width=1)), row=1, col=1)
    if "sma_50" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["sma_50"][-lookback:],
                     name="SMA50", line=dict(color="rgba(255,165,0,0.6)", width=1)), row=1, col=1)
    if "rsi" in df_feat:
        fig.add_trace(go.Scatter(x=df.index[-lookback:], y=df_feat["rsi"][-lookback:],
                     name="RSI", line=dict(color="#00d4aa")), row=2, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.4)", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,212,170,0.4)", row=2, col=1)
        fig.update_yaxes(range=[0, 100], row=2, col=1)
    fig.add_trace(go.Bar(x=df.index[-lookback:], y=df["volume"][-lookback:],
                 name="Vol", marker_color="rgba(255,255,255,0.15)"), row=3, col=1)
    fig.update_layout(
        height=480, template="plotly_dark",
        xaxis_rangeslider_visible=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="rgba(255,255,255,0.7)"),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, width='stretch')

    # Bottom row
    c_left, c_right = st.columns([1, 1])
    with c_left:
        st.markdown('<div class="section-header">Technical Indicators</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        if "rsi" in df_feat:
            rsi_v = df_feat["rsi"].iloc[-1]
            rsi_s = "Overbought" if rsi_v > 70 else "Oversold" if rsi_v < 30 else "Neutral"
            rsi_c = "#ff4444" if rsi_v > 70 else "#00d4aa" if rsi_v < 30 else "var(--text-dim)"
            cols[0].markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">RSI</div><div style="font-size:1.5rem;font-weight:700;color:{rsi_c}">{rsi_v:.1f}</div><small style="color:var(--text-dim)">{rsi_s}</small></div>', unsafe_allow_html=True)
        if "macd" in df_feat:
            macd_v = df_feat["macd"].iloc[-1]
            macd_sig = df_feat["macd_signal"].iloc[-1]
            macd_s = "Bullish" if macd_v > macd_sig else "Bearish"
            cols[1].markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">MACD</div><div style="font-size:1.5rem;font-weight:700">{macd_v:.2f}</div><small style="color:var(--text-dim)">{macd_s}</small></div>', unsafe_allow_html=True)
        if "bb_width" in df_feat:
            cols[2].markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Bollinger W</div><div style="font-size:1.5rem;font-weight:700">{df_feat["bb_width"].iloc[-1]:.2f}</div></div>', unsafe_allow_html=True)

        if models_exist(ticker):
            try:
                _, xgb_m, _, _, _, _ = load_models(ticker)
                if hasattr(xgb_m, "feature_importances_"):
                    imp = xgb_m.feature_importances_
                    feat_used = [c for c in FEATURE_COLS if c in df_feat.columns]
                    fi_df = pd.DataFrame({"f": feat_used, "i": imp}).sort_values("i", ascending=False).head(8)
                    fig_fi = go.Figure(go.Bar(x=fi_df["i"][::-1], y=fi_df["f"][::-1],
                        orientation="h", marker=dict(color=["#00d4aa" if v > fi_df["i"].mean() else "rgba(255,255,255,0.2)" for v in fi_df["i"][::-1]])))
                    fig_fi.update_layout(template="plotly_dark", height=260,
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=10, b=0), xaxis_visible=False, yaxis_title=None)
                    st.markdown('<div class="section-header" style="margin-top:1rem">Top Features</div>', unsafe_allow_html=True)
                    st.plotly_chart(fig_fi, width='stretch')
            except Exception:
                pass

    with c_right:
        st.markdown('<div class="section-header">Recent Data</div>', unsafe_allow_html=True)
        display_df = df.tail(7)[["open","high","low","close","volume"]].copy()
        display_df.index = display_df.index.strftime("%d %b")
        display_df.columns = [c.capitalize() for c in display_df.columns]
        display_df["Chg%"] = display_df["Close"].pct_change().mul(100).round(1).fillna(0)
        display_df["Close"] = display_df["Close"].round(2)
        display_df["Volume"] = (display_df["Volume"] / 1e6).round(1)
        st.dataframe(display_df.iloc[::-1], width='stretch')

        # Model details
        if ens_dir is not None and details:
            st.markdown('<div class="section-header" style="margin-top:1rem">Model Votes</div>', unsafe_allow_html=True)
            mc = st.columns(4)
            for i, (nm, k) in enumerate([("LSTM","lstm_dir"),("GRU","gru_dir"),("Transformer","transformer_dir"),("XGBoost","xgb_dir")]):
                d = details[k]
                c = "#00d4aa" if d == 1 else "#ff4444"
                a = "▲ UP" if d == 1 else "▼ DOWN"
                mc[i].markdown(f'<div class="glass" style="text-align:center;padding:0.6rem"><div class="metric-label">{nm}</div><div style="font-weight:600;color:{c}">{a}</div></div>', unsafe_allow_html=True)

    # Long term chart
    st.markdown('<div class="section-header" style="margin-top:0.5rem">Long Term Trend</div>', unsafe_allow_html=True)
    sma_200 = df["close"].rolling(200).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="rgba(255,255,255,0.8)", width=1)))
    fig2.add_trace(go.Scatter(x=df.index, y=df_feat["sma_50"], name="SMA 50", line=dict(color="rgba(124,58,237,0.6)", width=1)))
    fig2.add_trace(go.Scatter(x=df.index, y=sma_200, name="SMA 200", line=dict(color="rgba(255,165,0,0.6)", width=1)))
    fig2.update_layout(template="plotly_dark", height=300,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig2, width='stretch')


def tab_portfolio():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Portfolio</span></div>', unsafe_allow_html=True)
    p = st.session_state.portfolio
    stats = p.get_stats()
    if not stats:
        st.info("No trades. Go to **Backtest** to run a simulation.")
        return

    r1, r2, r3, r4 = st.columns(4)
    colr = "00d4aa" if stats["total_return"] >= 0 else "ff4444"
    r1.markdown(metric_card("Total Return", f'<span style="color:#{colr}">{stats["total_return"]:.1%}</span>'), unsafe_allow_html=True)
    r2.markdown(metric_card("Sharpe", stats["sharpe_ratio"]), unsafe_allow_html=True)
    r3.markdown(metric_card("Max DD", f'{stats["max_drawdown"]:.1f}%'), unsafe_allow_html=True)
    r4.markdown(metric_card("Win Rate", f'{stats["win_rate"]:.1f}%'), unsafe_allow_html=True)

    eq = pd.DataFrame(p.equity_curve)
    if len(eq) > 1:
        eq["date"] = pd.to_datetime(eq["date"])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
            fill="tozeroy", line=dict(color="#00d4aa", width=2),
            fillcolor="rgba(0,212,170,0.1)"))
        fig.update_layout(template="plotly_dark", height=350,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10,r=10,t=10,b=10), yaxis_title="Value (₹)")
        st.plotly_chart(fig, width='stretch')

    c1, c2 = st.columns(2)
    c1.markdown(metric_card("Cash", f'₹{stats["cash_remaining"]:,.2f}'), unsafe_allow_html=True)
    c2.markdown(metric_card("Holdings", f'₹{stats["holdings_value"]:,.2f}'), unsafe_allow_html=True)

    if p.trades:
        st.markdown(f'<div class="section-header" style="margin-top:1rem">Trade Log ({len(p.trades)})</div>', unsafe_allow_html=True)
        df_t = pd.DataFrame(p.trades).iloc[::-1]
        if "pnl" in df_t.columns:
            df_t["pnl"] = df_t["pnl"].round(2)
        st.dataframe(df_t, width='stretch', hide_index=True)


def tab_backtest():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Backtest</span></div>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock", NSE_STOCKS, key="bt_ticker")
    if not models_exist(ticker):
        st.warning("Train models first (Predictions tab)")
        return
    capital = st.number_input("Initial Capital (₹)", 10000, 10_000_000, 100000, step=50000)

    if st.button("▶ Run Backtest", type="primary", width='stretch'):
        with st.spinner("Running backtest..."):
            df = fetch_stock_data(ticker, period="5y")
            df_feat = add_technical_indicators(df)
            try:
                lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)
            except Exception as e:
                st.error(f"Failed to load models: {e}")
                st.stop()
            signals = generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feat)
            stats, portfolio = run_backtest(df_feat, signals, capital)
            st.session_state.portfolio = portfolio

        if stats:
            r1, r2, r3, r4 = st.columns(4)
            colr = "00d4aa" if stats["total_return"] >= 0 else "ff4444"
            r1.markdown(metric_card("Return", f'<span style="color:#{colr}">{stats["total_return"]:.1%}</span>'), unsafe_allow_html=True)
            r2.markdown(metric_card("Sharpe", stats["sharpe_ratio"]), unsafe_allow_html=True)
            r3.markdown(metric_card("Max DD", f'{stats["max_drawdown"]:.1f}%'), unsafe_allow_html=True)
            r4.markdown(metric_card("Win Rate", f'{stats["win_rate"]:.1f}%'), unsafe_allow_html=True)

            eq = pd.DataFrame(portfolio.equity_curve)
            if len(eq) > 1:
                eq["date"] = pd.to_datetime(eq["date"])
                bh = capital * (df["close"] / df["close"].iloc[0])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
                    name="Strategy", line=dict(color="#00d4aa", width=2)))
                bhd = df.index[:len(eq)]
                fig.add_trace(go.Scatter(x=bhd, y=bh.values[:len(bhd)], name="Buy & Hold",
                    line=dict(color="rgba(255,255,255,0.3)", width=1, dash="dash")))
                fig.update_layout(template="plotly_dark", height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10,r=10,t=10,b=10),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, width='stretch')

            df_t = pd.DataFrame(portfolio.trades)
            st.markdown(f'<div class="section-header">Trades ({len(df_t)})</div>', unsafe_allow_html=True)
            st.dataframe(df_t.iloc[::-1], width='stretch')


def tab_scanner():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Scanner</span></div>', unsafe_allow_html=True)
    trained = [t for t in NSE_STOCKS if models_exist(t)]
    st.markdown(f'Scanning <b>{len(trained)}</b> of {len(NSE_STOCKS)} NSE stocks', unsafe_allow_html=True)
    if not trained:
        st.warning("No trained models found.")
        return

    if st.button("🔍 Scan Now", type="primary", width='stretch'):
        results = []
        prog = st.progress(0)
        for i, ticker in enumerate(trained):
            try:
                df = fetch_stock_data(ticker, period="6mo")
                df_feat = add_technical_indicators(df)
                lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)
                d, conf, det = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat)
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
        c1.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Buy Signals</div><div style="font-size:1.8rem;font-weight:700;color:#00d4aa">{len(buys)}</div></div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Sell Signals</div><div style="font-size:1.8rem;font-weight:700;color:#ff4444">{len(sells)}</div></div>', unsafe_allow_html=True)
        st.dataframe(df_r, width='stretch', hide_index=True)


def tab_sentiment():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">News Sentiment</span></div>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock", NSE_STOCKS, key="sent_ticker")
    td = ticker.replace(".NS", "")
    if st.button(f"Analyze {td} News", type="primary", width='stretch'):
        with st.spinner(f"Analyzing news for {td}..."):
            try:
                score = fetch_news_sentiment(ticker)
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
                return
        lbl = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        clr = "#00d4aa" if score > 0.05 else "#ff4444" if score < -0.05 else "var(--text-dim)"
        st.markdown(f'<div class="glass" style="text-align:center;padding:2rem;"><div class="metric-label">Overall Sentiment</div><div style="font-size:3rem;font-weight:800;color:{clr};margin:0.5rem 0;">{score:+.3f}</div><span class="tag {"tag-up" if score>0.05 else "tag-down" if score<-0.05 else "tag-neutral"}">{lbl}</span></div>', unsafe_allow_html=True)
        st.caption("Based on latest headlines. Range: -1 (negative) to +1 (positive).")


def main():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;"><span style="font-size:2rem;">⚡</span><span class="gradient-text" style="font-size:1.6rem;font-weight:800;">StoMar</span><span style="font-size:0.75rem;color:var(--text-dim);margin-left:auto;">{datetime.now().strftime("%d %b %Y")}</span></div>', unsafe_allow_html=True)
    tabs = st.tabs(["📈 Predictions", "💰 Portfolio", "🔬 Backtest", "🔍 Scanner", "📰 Sentiment"])
    with tabs[0]: tab_predictions()
    with tabs[1]: tab_portfolio()
    with tabs[2]: tab_backtest()
    with tabs[3]: tab_scanner()
    with tabs[4]: tab_sentiment()
    st.divider()
    st.caption("⚠️ Educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()

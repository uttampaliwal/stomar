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
from src.backtester import run_simple_backtest as run_backtest, generate_model_signals, run_walk_forward_backtest, compute_metrics
from src.sentiment import get_stock_sentiment
from src.flow import get_flow_sentiment, fetch_options_pcr
from src.multitimeframe import fetch_mtf_data, get_combined_signal
from src.risk import generate_risk_report, kelly_criterion, calculate_var, calculate_cvar
from src.optimizer import optimize_portfolio
from src.holdings import parse_holdings_csv, compute_portfolio_stats
from src.regime import detect_regime

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
                _, xgb_m, _, _, _, _, _ = load_models(ticker)
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
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Walk-Forward Backtest</span></div>', unsafe_allow_html=True)
    st.caption("Out-of-sample testing: trains on 3y, tests on 1y, rolls forward — no data leakage")
    ticker = st.selectbox("Stock", NSE_STOCKS, key="bt_ticker")
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
            st.markdown('<div class="section-header">Walk-Forward Results (Out-of-Sample)</div>', unsafe_allow_html=True)
            r1, r2, r3, r4, r5 = st.columns(5)
            acc = metrics.get("ensemble_accuracy", 0)
            acc_c = "#00d4aa" if acc > 0.5 else "#ff4444"
            r1.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Ensemble Accuracy</div><div style="font-size:1.5rem;font-weight:700;color:{acc_c}">{acc:.1%}</div></div>', unsafe_allow_html=True)

            sim_ret = metrics.get("simulated_annual_return", 0)
            ret_c = "#00d4aa" if sim_ret > 0 else "#ff4444"
            r2.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Simulated Annual</div><div style="font-size:1.5rem;font-weight:700;color:{ret_c}">{sim_ret:.1%}</div></div>', unsafe_allow_html=True)

            r3.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Simulated Sharpe</div><div style="font-size:1.5rem;font-weight:700">{metrics.get("simulated_sharpe", 0):.2f}</div></div>', unsafe_allow_html=True)

            r4.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Test Days</div><div style="font-size:1.5rem;font-weight:700">{metrics.get("total_test_days", 0)}</div></div>', unsafe_allow_html=True)

            r5.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Windows</div><div style="font-size:1.5rem;font-weight:700">{metrics.get("n_windows", 0)}</div></div>', unsafe_allow_html=True)

            if results:
                st.markdown('<div class="section-header">Prediction Distribution</div>', unsafe_allow_html=True)
                rdf = pd.DataFrame(results)
                fig = go.Figure()
                buys = rdf[rdf["predicted"] == 1]
                sells = rdf[rdf["predicted"] == 0]
                fig.add_trace(go.Histogram(x=buys["ensemble_prob"], name="BUY", marker_color="#00d4aa", opacity=0.7))
                fig.add_trace(go.Histogram(x=sells["ensemble_prob"], name="SELL", marker_color="#ff4444", opacity=0.7))
                fig.update_layout(template="plotly_dark", height=300, barmode="overlay",
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10,r=10,t=10,b=10), xaxis_title="Ensemble Probability")
                st.plotly_chart(fig, width='stretch')

                acc_by_model = {}
                for m in ["lstm", "gru", "transformer"]:
                    correct = (rdf[m] == rdf["actual"]).sum()
                    acc_by_model[m.upper()] = correct / len(rdf) * 100
                xgb_correct = ((rdf["xgb_prob"] > 0.5).astype(int) == rdf["actual"]).sum()
                acc_by_model["XGBoost"] = xgb_correct / len(rdf) * 100

                st.markdown('<div class="section-header">Individual Model Accuracy</div>', unsafe_allow_html=True)
                mc = st.columns(4)
                for i, (nm, a) in enumerate(acc_by_model.items()):
                    c = "#00d4aa" if a > 50 else "#ff4444"
                    mc[i].markdown(f'<div class="glass" style="text-align:center;padding:0.6rem"><div class="metric-label">{nm}</div><div style="font-size:1.2rem;font-weight:700;color:{c}">{a:.1f}%</div></div>', unsafe_allow_html=True)

                st.markdown(f'<div class="section-header">Per-Window Breakdown</div>', unsafe_allow_html=True)
                window_size = len(results) // metrics.get("n_windows", 1)
                for w in range(metrics.get("n_windows", 0)):
                    start = w * window_size
                    end = min(start + window_size, len(results))
                    window_results = results[start:end]
                    if window_results:
                        wr = sum(1 for r in window_results if r["predicted"] == r["actual"]) / len(window_results) * 100
                        first_date = window_results[0]["date"]
                        last_date = window_results[-1]["date"]
                        bar_c = "#00d4aa" if wr > 50 else "#ff4444"
                        st.markdown(f'<div style="display:flex;align-items:center;gap:1rem;margin:0.3rem 0;"><span style="color:var(--text-dim);min-width:200px;font-size:0.85rem">{first_date} → {last_date}</span><div style="flex:1;height:8px;background:var(--card-border);border-radius:4px;overflow:hidden;"><div style="width:{wr}%;height:100%;background:{bar_c};border-radius:4px;"></div></div><span style="font-weight:600;min-width:50px;text-align:right;">{wr:.1f}%</span></div>', unsafe_allow_html=True)
        else:
            st.warning("Not enough data for walk-forward backtest.")


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
                result = get_stock_sentiment(ticker)
                score = result.get("score", 0.0)
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")
                return
        lbl = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"
        clr = "#00d4aa" if score > 0.05 else "#ff4444" if score < -0.05 else "var(--text-dim)"
        st.markdown(f'<div class="glass" style="text-align:center;padding:2rem;"><div class="metric-label">Overall Sentiment</div><div style="font-size:3rem;font-weight:800;color:{clr};margin:0.5rem 0;">{score:+.3f}</div><span class="tag {"tag-up" if score>0.05 else "tag-down" if score<-0.05 else "tag-neutral"}">{lbl}</span></div>', unsafe_allow_html=True)
        st.caption("Based on latest headlines. Range: -1 (negative) to +1 (positive).")


def tab_market_pulse():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Market Pulse</span></div>', unsafe_allow_html=True)
    st.caption("Institutional flows, options PCR, and multi-timeframe signals")

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
                fii_c = "#00d4aa" if fii_net >= 0 else "#ff4444"
                dii_c = "#00d4aa" if dii_net >= 0 else "#ff4444"
                fii_sign = "+" if fii_net >= 0 else ""
                dii_sign = "+" if dii_net >= 0 else ""
                st.markdown(f'''<div class="glass">
                    <div style="display:flex;gap:1.5rem;justify-content:center;">
                        <div style="text-align:center"><div class="metric-label">FII</div><div style="font-size:1.5rem;font-weight:700;color:{fii_c}">{fii_sign}₹{abs(fii_net):,.0f} Cr</div><small style="color:var(--text-dim)">Buy ₹{row["fii_buy"]:,.0f} / Sell ₹{row["fii_sell"]:,.0f}</small></div>
                        <div style="text-align:center"><div class="metric-label">DII</div><div style="font-size:1.5rem;font-weight:700;color:{dii_c}">{dii_sign}₹{abs(dii_net):,.0f} Cr</div><small style="color:var(--text-dim)">Buy ₹{row["dii_buy"]:,.0f} / Sell ₹{row["dii_sell"]:,.0f}</small></div>
                    </div>
                    <div style="text-align:center;margin-top:0.8rem;"><span class="tag {'tag-up' if 'Bull' in flow_sentiment else 'tag-down' if 'Bear' in flow_sentiment else 'tag-neutral'}">{flow_sentiment}</span></div>
                </div>''', unsafe_allow_html=True)
            else:
                st.markdown('<div class="glass" style="text-align:center;padding:1.5rem;"><div class="metric-label">FII / DII Data</div><div style="margin-top:0.5rem;color:var(--text-dim)">No data available</div></div>', unsafe_allow_html=True)
        except Exception as e:
            st.markdown(f'<div class="glass" style="text-align:center;padding:1.5rem;"><div class="metric-label">FII / DII Data</div><div style="margin-top:0.5rem;color:var(--text-dim)">Unavailable: {e}</div></div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="section-header">Options PCR & Max Pain</div>', unsafe_allow_html=True)
        pcr = fetch_options_pcr()
        if pcr.get("pcr_oi", 0) > 0:
            pcr_v = pcr["pcr_oi"]
            pcr_c = "#00d4aa" if pcr_v > 1 else "#ff4444" if pcr_v < 0.8 else "var(--text-dim)"
            pcr_lbl = "Bullish" if pcr_v > 1 else "Bearish" if pcr_v < 0.8 else "Neutral"
            st.markdown(f'''<div class="glass">
                <div style="text-align:center"><div class="metric-label">Put-Call Ratio ({pcr.get('symbol','NIFTY')})</div><div style="font-size:2rem;font-weight:700;color:{pcr_c}">{pcr_v:.3f}</div><span class="tag {'tag-up' if pcr_v>1 else 'tag-down' if pcr_v<0.8 else 'tag-neutral'}">{pcr_lbl}</span></div>
                <div style="display:flex;gap:1.5rem;justify-content:center;margin-top:0.8rem;">
                    <div style="text-align:center"><div class="metric-label">Call OI</div><div style="font-weight:600">{pcr["call_oi"]/1e6:.1f}M</div></div>
                    <div style="text-align:center"><div class="metric-label">Put OI</div><div style="font-weight:600">{pcr["put_oi"]/1e6:.1f}M</div></div>
                </div>
                <div style="text-align:center;margin-top:0.6rem;"><div class="metric-label">Max Pain</div><div style="font-weight:600">₹{pcr['max_pain']:,.0f}</div></div>
            </div>''', unsafe_allow_html=True)
        else:
            st.markdown('<div class="glass" style="text-align:center;padding:1.5rem;"><div class="metric-label">Options Data</div><div style="margin-top:0.5rem;color:var(--text-dim)">Market closed — PCR only available during trading hours</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-header" style="margin-top:1rem">Multi-Timeframe Analysis</div>', unsafe_allow_html=True)
    tf_ticker = st.selectbox("Stock", NSE_STOCKS, key="tf_ticker")
    if st.button("Analyze Timeframes", type="primary", key="tf_btn", width='stretch'):
        with st.spinner(f"Analyzing {tf_ticker} across 4 timeframes..."):
            mtf = fetch_mtf_data(tf_ticker)
            signal = get_combined_signal(mtf)

        if signal["timeframes"]:
            sig_c = "#00d4aa" if signal["signal"] == "Bullish" else "#ff4444" if signal["signal"] == "Bearish" else "var(--text-dim)"
            st.markdown(f'''<div class="glass" style="text-align:center;padding:1.5rem;">
                <div class="metric-label">Combined Signal ({tf_ticker.replace('.NS','')})</div>
                <div style="font-size:2rem;font-weight:800;color:{sig_c};margin:0.5rem 0;">{signal['signal']}</div>
                <span class="tag {'tag-up' if signal['signal']=='Bullish' else 'tag-down' if signal['signal']=='Bearish' else 'tag-neutral'}">Confidence: {signal['confidence']:.0f}%</span>
            </div>''', unsafe_allow_html=True)

            tf_cols = st.columns(4)
            tf_names = {"15m": "15 Minute", "1h": "1 Hour", "daily": "Daily", "weekly": "Weekly"}
            for i, (tf, name) in enumerate(tf_names.items()):
                if tf in signal["timeframes"]:
                    sig = signal["timeframes"][tf]
                    tc = "#00d4aa" if sig["signal"] == "Bullish" else "#ff4444" if sig["signal"] == "Bearish" else "var(--text-dim)"
                    ind_str = "<br>".join([f'<small>{k}: {v}</small>' for k, v in sig["indicators"].items()])
                    tf_cols[i].markdown(f'''<div class="glass" style="text-align:center;padding:0.8rem;">
                        <div class="metric-label">{name}</div>
                        <div style="font-size:1.2rem;font-weight:700;color:{tc};margin:0.3rem 0;">{sig['signal']}</div>
                        <small style="color:var(--text-dim)">Strength: {sig['strength']:.0%}</small>
                        <div style="margin-top:0.5rem;">{ind_str}</div>
                    </div>''', unsafe_allow_html=True)
        else:
            st.info("No timeframe data available.")


def tab_optimizer():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Portfolio Optimizer</span></div>', unsafe_allow_html=True)
    st.caption("Mean-Variance Optimization + Black-Litterman — finds optimal allocation")

    selected = st.multiselect("Select Stocks", NSE_STOCKS, default=NSE_STOCKS[:3], key="opt_stocks")
    if len(selected) < 2:
        st.info("Select at least 2 stocks")
        return

    period = st.selectbox("History", ["6mo", "1y", "2y", "5y"], index=2, key="opt_period")

    if st.button("Optimize Portfolio", type="primary", width='stretch', key="opt_btn"):
        with st.spinner("Fetching prices and optimizing..."):
            prices = pd.DataFrame()
            for t in selected:
                df = fetch_stock_data(t, period=period)
                prices[t] = df["close"]
            prices = prices.dropna()

            if len(prices) < 60:
                st.warning("Not enough price history")
                return

            result = optimize_portfolio(prices)

        st.markdown('<div class="section-header">Optimal Portfolios</div>', unsafe_allow_html=True)
        for ptype, label in [("max_sharpe", "Max Sharpe"), ("min_variance", "Min Variance"), ("black_litterman", "Black-Litterman")]:
            p = result[ptype]
            c1, c2, c3 = st.columns([2, 1, 1])
            with c1:
                st.markdown(f"**{label}**")
                for i, t in enumerate(selected):
                    w = p["weights"][i]
                    bar = "█" * int(w * 30)
                    st.markdown(f"  {t.replace('.NS','')}: **{w:.1%}** {bar}")
            with c2:
                st.metric("Exp. Return", f"{p['return']:.1%}")
            with c3:
                st.metric("Volatility", f"{p['volatility']:.1%}")
            st.divider()

        if result["efficient_frontier"]:
            ef = pd.DataFrame(result["efficient_frontier"])
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ef["volatility"], y=ef["return"], mode="lines+markers",
                name="Efficient Frontier", line=dict(color="#00d4aa", width=2)))
            ms = result["max_sharpe"]
            fig.add_trace(go.Scatter(x=[ms["volatility"]], y=[ms["return"]], mode="markers",
                name="Max Sharpe", marker=dict(color="#7c3aed", size=15, symbol="star")))
            fig.update_layout(template="plotly_dark", height=400,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                xaxis_title="Volatility", yaxis_title="Return",
                margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig, width='stretch')


def tab_holdings():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Holdings Tracker</span></div>', unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Zerodha Holdings CSV", type=["csv"], key="holdings_upload")

    if uploaded:
        holdings_df = parse_holdings_csv(uploaded)
        stats = compute_portfolio_stats(holdings_df)

        if stats:
            r1, r2, r3, r4 = st.columns(4)
            colr = "00d4aa" if stats["total_pnl"] >= 0 else "ff4444"
            r1.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Invested</div><div style="font-size:1.3rem;font-weight:700">₹{stats["total_invested"]:,.0f}</div></div>', unsafe_allow_html=True)
            r2.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Current</div><div style="font-size:1.3rem;font-weight:700">₹{stats["total_current"]:,.0f}</div></div>', unsafe_allow_html=True)
            r3.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">P&L</div><div style="font-size:1.3rem;font-weight:700;color:#{colr}">₹{stats["total_pnl"]:+,.0f}</div></div>', unsafe_allow_html=True)
            r4.markdown(f'<div class="glass" style="text-align:center"><div class="metric-label">Return</div><div style="font-size:1.3rem;font-weight:700;color:#{colr}">{stats["total_return_pct"]:+.1f}%</div></div>', unsafe_allow_html=True)

            st.markdown('<div class="section-header">Category Allocation</div>', unsafe_allow_html=True)
            cat_data = []
            for cat, info in sorted(stats["categories"].items(), key=lambda x: -x[1]["weight"]):
                cat_data.append({"Category": cat, "Value": f"₹{info['value']:,.0f}", "Weight": f"{info['weight']:.1%}", "Return": f"{info['return_pct']:+.1f}%"})
            st.dataframe(pd.DataFrame(cat_data), width='stretch', hide_index=True)

            if stats["categories"]:
                cat_labels = list(stats["categories"].keys())
                cat_values = [stats["categories"][c]["value"] for c in cat_labels]
                fig = go.Figure(go.Pie(labels=cat_labels, values=cat_values,
                    hole=0.4, marker=dict(colors=px.colors.qualitative.Set3)))
                fig.update_layout(template="plotly_dark", height=400,
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=10,r=10,t=10,b=10), showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2))
                st.plotly_chart(fig, width='stretch')

            st.markdown('<div class="section-header">Holdings Detail</div>', unsafe_allow_html=True)
            h_data = []
            for h in stats["holdings"]:
                ret_c = "#00d4aa" if h["return_pct"] >= 0 else "#ff4444"
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
                risk_notes.append("High concentration (Herfindahl > 0.25)")
            if top3_weight > 0.60:
                risk_notes.append(f"Top 3 holdings = {top3_weight:.0%} of portfolio")
            if any(h["return_pct"] < -5 for h in stats["holdings"]):
                risk_notes.append("Some holdings with >5% loss")
            if not risk_notes:
                risk_notes.append("Portfolio risk profile looks reasonable")
            for note in risk_notes:
                st.warning(note)


def tab_risk():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:1rem;"><span class="gradient-text" style="font-size:2rem;font-weight:800;">Risk Dashboard</span></div>', unsafe_allow_html=True)

    ticker = st.selectbox("Stock", NSE_STOCKS, key="risk_ticker")
    period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=2, key="risk_period")

    if st.button("Analyze Risk", type="primary", width='stretch', key="risk_btn"):
        with st.spinner("Computing risk metrics..."):
            df = fetch_stock_data(ticker, period=period)
            returns = df["close"].pct_change().dropna().values
            equity = (1 + returns).cumprod() * 100000
            report = generate_risk_report(returns, equity)

            regime = detect_regime(df["close"])

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="section-header">Market Regime</div>', unsafe_allow_html=True)
            regime_c = "#00d4aa" if regime["regime"] == "Bull" else "#ff4444" if regime["regime"] == "Bear" else "#ffa500"
            st.markdown(f'<div class="glass" style="text-align:center;padding:1.5rem;"><div class="metric-label">Current Regime</div><div style="font-size:2rem;font-weight:800;color:{regime_c};margin:0.5rem 0;">{regime["regime"]}</div><span class="tag {"tag-up" if regime["regime"]=="Bull" else "tag-down" if regime["regime"]=="Bear" else "tag-neutral"}">{regime["confidence"]:.0f}% confidence</span></div>', unsafe_allow_html=True)
            rec = regime["recommendation"]
            st.markdown(f'<div class="glass"><div class="metric-label">Recommendation</div><div style="font-weight:600;margin:0.3rem 0;">{rec["action"]}</div><small>Allocation: {rec["allocation"]}</small><br><small>Risk: {rec["risk_level"]}</small></div>', unsafe_allow_html=True)

        with c2:
            st.markdown('<div class="section-header">Risk Metrics</div>', unsafe_allow_html=True)
            metrics_grid = [
                ("Annual Return", f"{report['annual_return']:.1f}%", "#00d4aa" if report['annual_return'] > 0 else "#ff4444"),
                ("Annual Volatility", f"{report['annual_volatility']:.1f}%", "var(--text-dim)"),
                ("Sharpe Ratio", f"{report['sharpe']:.3f}", "#00d4aa" if report['sharpe'] > 1 else "#ff4444"),
                ("Sortino Ratio", f"{report['sortino']:.3f}", "#00d4aa" if report['sortino'] > 1 else "#ff4444"),
                ("Max Drawdown", f"{report['max_drawdown']:.1f}%", "#ff4444"),
                ("VaR (95%)", f"{report['var_95']:.2f}%", "#ff4444"),
                ("CVaR (95%)", f"{report['cvar_95']:.2f}%", "#ff4444"),
                ("Calmar Ratio", f"{report['calmar']:.3f}", "#00d4aa" if report['calmar'] > 1 else "#ffa500"),
                ("Win Days", f"{report['positive_days_pct']:.0f}%", "#00d4aa" if report['positive_days_pct'] > 50 else "#ff4444"),
                ("Best Day", f"{report['best_day']:.2f}%", "#00d4aa"),
                ("Worst Day", f"{report['worst_day']:.2f}%", "#ff4444"),
                ("Skewness", f"{report['skewness']:.2f}", "var(--text-dim)"),
            ]
            for i in range(0, len(metrics_grid), 3):
                cols = st.columns(3)
                for j in range(3):
                    if i+j < len(metrics_grid):
                        label, val, color = metrics_grid[i+j]
                        cols[j].markdown(f'<div class="glass" style="text-align:center;padding:0.6rem"><div class="metric-label">{label}</div><div style="font-size:1.1rem;font-weight:700;color:{color}">{val}</div></div>', unsafe_allow_html=True)

        st.markdown('<div class="section-header">Drawdown Chart</div>', unsafe_allow_html=True)
        cummax = np.maximum.accumulate(equity)
        drawdown = (equity - cummax) / cummax * 100
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index[-len(drawdown):], y=drawdown, fill="tozeroy",
            line=dict(color="#ff4444", width=1), fillcolor="rgba(255,68,68,0.2)", name="Drawdown"))
        fig.update_layout(template="plotly_dark", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            yaxis_title="Drawdown %", margin=dict(l=10,r=10,t=10,b=10))
        st.plotly_chart(fig, width='stretch')

        st.markdown('<div class="section-header">Kelly Criterion</div>', unsafe_allow_html=True)
        win_days = (returns > 0).sum()
        lose_days = (returns < 0).sum()
        avg_win = returns[returns > 0].mean() if win_days > 0 else 0
        avg_loss = abs(returns[returns < 0].mean()) if lose_days > 0 else 0.001
        win_rate = win_days / len(returns)
        kelly = kelly_criterion(win_rate, avg_win, avg_loss)
        st.markdown(f'<div class="glass"><div class="metric-label">Optimal Position Size (Kelly)</div><div style="font-size:1.5rem;font-weight:700">{kelly:.1%}</div><small>Win rate: {win_rate:.1%} | Avg win: {avg_win:.3f} | Avg loss: {avg_loss:.3f}</small></div>', unsafe_allow_html=True)


def main():
    st.markdown(f'<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;"><span style="font-size:2rem;">⚡</span><span class="gradient-text" style="font-size:1.6rem;font-weight:800;">StoMar</span><span style="font-size:0.75rem;color:var(--text-dim);margin-left:auto;">{datetime.now().strftime("%d %b %Y")}</span></div>', unsafe_allow_html=True)
    tabs = st.tabs(["📈 Predictions", "💰 Portfolio", "🔬 Backtest", "🔍 Scanner", "📰 Sentiment", "🏛️ Market Pulse", "📊 Optimizer", "💼 Holdings", "⚡ Risk"])
    with tabs[0]: tab_predictions()
    with tabs[1]: tab_portfolio()
    with tabs[2]: tab_backtest()
    with tabs[3]: tab_scanner()
    with tabs[4]: tab_sentiment()
    with tabs[5]: tab_market_pulse()
    with tabs[6]: tab_optimizer()
    with tabs[7]: tab_holdings()
    with tabs[8]: tab_risk()
    st.divider()
    st.caption("⚠️ Educational purposes only. Not financial advice.")

if __name__ == "__main__":
    main()

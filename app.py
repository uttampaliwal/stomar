import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os, sys, traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import fetch_stock_data, NSE_STOCKS, get_market_status
from src.features import add_technical_indicators
from src.model import load_models, models_exist, DEVICE
from src.trainer import train_for_ticker, FEATURE_COLS
from src.ensemble import predict_ensemble, backtest_ensemble
from src.sentiment import fetch_news_sentiment
from src.portfolio import Portfolio
from src.backtester import run_backtest, generate_model_signals

st.set_page_config(page_title="StoMar - Trading Suite", page_icon="📈", layout="wide")

st.markdown("""
<style>
    .main-header {font-size: 2.2rem; font-weight: 700;}
    .pred-up {color: #00ff88; font-weight: 700; font-size: 1.3rem;}
    .pred-down {color: #ff4444; font-weight: 700; font-size: 1.3rem;}
    .metric-box {background: #1e1e1e; padding: 1rem; border-radius: 10px; border: 1px solid #333;}
    .green {color: #00ff88;}
    .red {color: #ff4444;}
</style>
""", unsafe_allow_html=True)

if "portfolio" not in st.session_state:
    st.session_state.portfolio = Portfolio(initial_capital=100000)
if "last_ticker" not in st.session_state:
    st.session_state.last_ticker = NSE_STOCKS[0]


@st.cache_data(ttl=3600)
def get_data(ticker, period):
    return fetch_stock_data(ticker, period=period)


def page_predictions():
    st.markdown('<p class="main-header">📈 StoMar Predictions</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns([2, 1, 1, 0.8])
    with col1:
        ticker = st.selectbox("Stock", NSE_STOCKS, index=NSE_STOCKS.index(st.session_state.last_ticker))
        st.session_state.last_ticker = ticker
    with col2:
        period = st.selectbox("Period", ["6mo", "1y", "2y", "5y"], index=3)
    with col3:
        st.markdown("###")
        refresh = st.button("🔄 Refresh")
    with col4:
        st.markdown("###")
        train_btn = st.button("⚡ Train")

    market = get_market_status()
    st.caption(f"Market: {'🟢' if market=='Open' else '🔴'} {market}")

    try:
        df = get_data(ticker, period)
    except Exception as e:
        st.error(f"Data fetch failed: {e}")
        st.stop()

    ticker_display = ticker.replace(".NS", "")
    df_feat = add_technical_indicators(df)

    # Price row
    curr = df["close"].iloc[-1]
    prev = df["close"].iloc[-2]
    chg = curr - prev
    chg_pct = (chg / prev) * 100
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"₹{curr:.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
    c2.metric("Range", f"₹{df['low'].iloc[-1]:.2f} – ₹{df['high'].iloc[-1]:.2f}")
    c3.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")

    # Prediction
    if train_btn:
        with st.spinner(f"Training {ticker_display} (~3 min)..."):
            r = train_for_ticker(ticker, force_retrain=True)
            st.success(f"XGB: {r['xgb_accuracy']:.1%} | LSTM: {r['lstm_accuracy']:.1%} | GRU: {r['gru_accuracy']:.1%} | TF: {r['transformer_accuracy']:.1%} | Ensemble: {r['ensemble_accuracy']:.1%}")
            st.cache_data.clear()

    if models_exist(ticker):
        lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)
        ens_dir, conf, details = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat)
        if ens_dir is not None:
            icon = "📈" if ens_dir == 1 else "📉"
            cls = "pred-up" if ens_dir == 1 else "pred-down"
            c4.markdown(f"**Ensemble**<br><span class='{cls}'>{icon} {'UP' if ens_dir==1 else 'DOWN'}</span><br><small>{conf:.1f}% confidence</small>", unsafe_allow_html=True)

            # Model breakdown
            with st.expander("⚙️ Model Details", expanded=False):
                cols = st.columns(4)
                for i, (name, key) in enumerate([("LSTM","lstm_dir"), ("GRU","gru_dir"), ("Transformer","transformer_dir"), ("XGBoost","xgb_dir")]):
                    d = details[key]
                    txt = "📈 UP" if d == 1 else "📉 DOWN"
                    cols[i].metric(name, txt)
                st.metric("XGBoost UP Probability", f"{details['xgb_prob_up']:.1%}")
                st.metric("Ensemble Probability", f"{details['ensemble_prob']:.1%}")
        else:
            c4.metric("Prediction", "Need 60d data")
    else:
        c4.metric("Prediction", "Not trained")
        if st.button(f"Train {ticker_display}"):
            with st.spinner("Training..."):
                r = train_for_ticker(ticker, force_retrain=True)
                st.success(f"Ensemble: {r['ensemble_accuracy']:.1%}")
                st.cache_data.clear()
                st.rerun()

    # Chart
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=(f"{ticker_display} Price", "RSI", "Volume"))
    fig.add_trace(go.Candlestick(x=df.index[-90:], open=df["open"][-90:], high=df["high"][-90:],
                                 low=df["low"][-90:], close=df["close"][-90:], name=""), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-90:], y=df_feat["sma_20"][-90:], name="SMA 20",
                             line=dict(color="orange", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-90:], y=df_feat["sma_50"][-90:], name="SMA 50",
                             line=dict(color="purple", width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index[-90:], y=df_feat["rsi"][-90:], name="RSI",
                             line=dict(color="cyan")), row=2, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(go.Bar(x=df.index[-90:], y=df["volume"][-90:], name="Vol",
                         marker_color="gray"), row=3, col=1)
    fig.update_layout(height=550, template="plotly_dark", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # Indicators
    with st.expander("📊 Technical Indicators", expanded=False):
        cols = st.columns(5)
        rsi_v = df_feat["rsi"].iloc[-1]
        cols[0].metric("RSI", f"{rsi_v:.1f}", "Overbought" if rsi_v>70 else "Oversold" if rsi_v<30 else "Neutral")
        macd_v = df_feat["macd"].iloc[-1]
        macd_s = df_feat["macd_signal"].iloc[-1]
        cols[1].metric("MACD", f"{macd_v:.2f}", "Bullish" if macd_v>macd_s else "Bearish")
        cols[2].metric("Bollinger Width", f"{df_feat['bb_width'].iloc[-1]:.2f}")
        cols[3].metric("ATR", f"{df_feat['atr'].iloc[-1]:.2f}")
        cols[4].metric("OBV", f"{df_feat['obv'].iloc[-1]:,.0f}")

    # Feature importance
    if models_exist(ticker):
        _, xgb_m, _, _, _, _ = load_models(ticker)
        if hasattr(xgb_m, "feature_importances_"):
            imp = xgb_m.feature_importances_
            feat_used = [c for c in FEATURE_COLS if c in df_feat.columns]
            fi_df = pd.DataFrame({"f": feat_used, "i": imp}).sort_values("i", ascending=True).tail(12)
            fig_fi = go.Figure(go.Bar(x=fi_df["i"], y=fi_df["f"], orientation="h", marker_color="limegreen"))
            fig_fi.update_layout(template="plotly_dark", height=300, margin=dict(l=0, r=0, t=0, b=0))
            with st.expander("🔑 Feature Importance", expanded=False):
                st.plotly_chart(fig_fi, use_container_width=True)


def page_portfolio():
    st.markdown('<p class="main-header">💰 Portfolio Tracker</p>', unsafe_allow_html=True)
    p = st.session_state.portfolio
    stats = p.get_stats()

    if not stats:
        st.info("No trades yet. Go to Backtest tab to run a simulation.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Return", f"{stats['total_return']:.1%}")
    c2.metric("Sharpe Ratio", stats["sharpe_ratio"])
    c3.metric("Max Drawdown", f"{stats['max_drawdown']:.1f}%")
    c4.metric("Win Rate", f"{stats['win_rate']:.1f}%")

    eq = pd.DataFrame(p.equity_curve)
    if len(eq) > 1:
        eq["date"] = pd.to_datetime(eq["date"])
        fig = go.Figure(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
                                   fill="tozeroy", line=dict(color="limegreen")))
        fig.update_layout(template="plotly_dark", height=350, title="Equity Curve")
        st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    c1.metric("Cash", f"₹{stats['cash_remaining']:,.2f}")
    c2.metric("Holdings Value", f"₹{stats['holdings_value']:,.2f}")

    if p.trades:
        st.markdown("### Trade Log")
        df_t = pd.DataFrame(p.trades)
        cols = ["date", "ticker", "action", "price", "quantity"]
        if "pnl" in df_t.columns:
            cols.append("pnl")
        st.dataframe(df_t[cols].iloc[::-1], use_container_width=True)


def page_backtest():
    st.markdown('<p class="main-header">🔬 Strategy Backtest</p>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock to backtest", NSE_STOCKS, key="bt_ticker")

    if not models_exist(ticker):
        st.warning("Train models first (Predictions tab)")
        return

    capital = st.number_input("Initial Capital (₹)", 10000, 10000000, 100000, step=50000)

    if st.button("▶ Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            df = fetch_stock_data(ticker, period="5y")
            df_feat = add_technical_indicators(df)
            lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)

            signals = generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feat)
            stats, portfolio = run_backtest(df_feat, signals, capital)

            st.session_state.portfolio = portfolio

        if stats:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Return", f"{stats['total_return']:.1%}")
            c2.metric("Sharpe", stats["sharpe_ratio"])
            c3.metric("Max DD", f"{stats['max_drawdown']:.1f}%")
            c4.metric("Win Rate", f"{stats['win_rate']:.1f}%")

            eq = pd.DataFrame(portfolio.equity_curve)
            if len(eq) > 1:
                eq["date"] = pd.to_datetime(eq["date"])
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=eq["date"], y=eq["equity"], mode="lines",
                                         fill="tozeroy", line=dict(color="limegreen"), name="Strategy"))
                bh = capital * (df["close"] / df["close"].iloc[0])
                bh_dates = df.index[:len(eq)]
                if len(bh_dates) > len(eq):
                    bh_dates = bh_dates[:len(eq)]
                fig.add_trace(go.Scatter(x=bh_dates, y=bh.values[:len(bh_dates)],
                                         line=dict(color="gray", dash="dash"), name="Buy & Hold"))
                fig.update_layout(template="plotly_dark", height=400, title="Equity vs Buy & Hold")
                st.plotly_chart(fig, use_container_width=True)

            st.markdown("### Trade Log")
            df_t = pd.DataFrame(portfolio.trades)
            st.dataframe(df_t.iloc[::-1], use_container_width=True)
        else:
            st.error("Backtest failed: no signals generated")


def page_scanner():
    st.markdown('<p class="main-header">🔍 Market Scanner</p>', unsafe_allow_html=True)
    st.markdown("Scanning **all 20 NSE stocks** for signals...")
    trained = [t for t in NSE_STOCKS if models_exist(t)]

    if not trained:
        st.warning("No trained models found. Train some stocks first.")
        return

    results = []
    progress = st.progress(0)
    status = st.empty()

    for i, ticker in enumerate(trained):
        status.text(f"Scanning {ticker}...")
        try:
            df = fetch_stock_data(ticker, period="6mo")
            df_feat = add_technical_indicators(df)
            lstm, gru, transformer, xgb, scaler, feat = load_models(ticker)
            d, conf, details = predict_ensemble(lstm, gru, transformer, xgb, scaler, feat, df_feat)

            price = df["close"].iloc[-1]
            chg = (df["close"].iloc[-1] / df["close"].iloc[-5] - 1) * 100
            rsi = df_feat["rsi"].iloc[-1]

            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Signal": "📈 BUY" if d == 1 else "📉 SELL",
                "Confidence": f"{conf:.0f}%",
                "Price": f"₹{price:.1f}",
                "5d Change": f"{chg:+.1f}%",
                "RSI": f"{rsi:.0f}",
                "Score": conf * (1 if d == 1 else -1),
            })
        except Exception:
            results.append({
                "Ticker": ticker.replace(".NS", ""),
                "Signal": "❌ Error", "Confidence": "-",
                "Price": "-", "5d Change": "-", "RSI": "-", "Score": 0,
            })
        progress.progress((i + 1) / len(trained))

    status.text("")

    df_r = pd.DataFrame(results).sort_values("Score", ascending=False)
    df_r = df_r.drop(columns=["Score"])

    st.markdown(f"### Signals — {len(trained)} stocks scanned")
    st.dataframe(df_r, use_container_width=True)

    # Highlight top picks
    top_buy = [r["Ticker"] for r in results if r.get("Score", 0) > 50][:3]
    top_sell = [r["Ticker"] for r in results if r.get("Score", 0) < -50][:3]

    if top_buy:
        st.markdown(f"**Strongest BUY signals:** {', '.join(top_buy)}")
    if top_sell:
        st.markdown(f"**Strongest SELL signals:** {', '.join(top_sell)}")


def page_sentiment():
    st.markdown('<p class="main-header">📰 News Sentiment</p>', unsafe_allow_html=True)
    ticker = st.selectbox("Stock", NSE_STOCKS, key="sent_ticker")
    ticker_display = ticker.replace(".NS", "")

    if st.button("🔍 Analyze News Sentiment", type="primary"):
        with st.spinner(f"Analyzing latest news for {ticker_display}..."):
            score = fetch_news_sentiment(ticker)

        score_label = "Positive ✅" if score > 0.05 else "Negative ❌" if score < -0.05 else "Neutral ⚖️"
        color = "green" if score > 0.05 else "red" if score < -0.05 else "white"

        st.markdown(f"### Overall Sentiment: <span style='color:{color}'>{score_label}</span>", unsafe_allow_html=True)
        st.metric("Sentiment Score", f"{score:+.3f}")
        st.caption("Score range: -1 (very negative) to +1 (very positive). Based on latest headlines.")


def main():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        ["📈 Predictions", "💰 Portfolio", "🔬 Backtest", "🔍 Scanner", "📰 Sentiment"]
    )
    with tab1: page_predictions()
    with tab2: page_portfolio()
    with tab3: page_backtest()
    with tab4: page_scanner()
    with tab5: page_sentiment()

    st.divider()
    st.caption("⚠️ Educational purposes only. Not financial advice.")


if __name__ == "__main__":
    main()

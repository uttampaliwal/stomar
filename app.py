import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.data_fetcher import fetch_stock_data, NSE_STOCKS, get_market_status
from src.features import add_technical_indicators
from src.model import load_models, models_exist
from src.trainer import train_for_ticker

st.set_page_config(
    page_title="StoMar - Stock Market Predictor",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header {font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem;}
    .pred-up {color: #00ff88; font-weight: 600;}
    .pred-down {color: #ff4444; font-weight: 600;}
    .metric-card {background: #1e1e1e; padding: 1rem; border-radius: 10px; border: 1px solid #333;}
</style>
""",
    unsafe_allow_html=True,
)


def plot_candlestick(df: pd.DataFrame, ticker: str):
    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "RSI", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index[-90:],
            open=df["open"][-90:],
            high=df["high"][-90:],
            low=df["low"][-90:],
            close=df["close"][-90:],
            name="Price",
        ),
        row=1,
        col=1,
    )

    if "sma_20" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-90:], y=df["sma_20"][-90:], name="SMA 20", line=dict(color="orange")),
            row=1,
            col=1,
        )
    if "sma_50" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-90:], y=df["sma_50"][-90:], name="SMA 50", line=dict(color="purple")),
            row=1,
            col=1,
        )

    if "rsi" in df.columns:
        fig.add_trace(
            go.Scatter(x=df.index[-90:], y=df["rsi"][-90:], name="RSI", line=dict(color="cyan")),
            row=2,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    fig.add_trace(
        go.Bar(x=df.index[-90:], y=df["volume"][-90:], name="Volume", marker_color="gray"),
        row=3,
        col=1,
    )

    fig.update_layout(
        height=700,
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def prediction_page():
    st.markdown('<p class="main-header">StoMar 📈</p>', unsafe_allow_html=True)
    st.markdown("### Stock Market Prediction for NSE India")

    market_status = get_market_status()
    status_color = "🟢" if market_status == "Open" else "🔴"
    st.markdown(f"**Market Status:** {status_color} {market_status}")

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        ticker = st.selectbox("Select Stock", NSE_STOCKS, index=0)
    with col2:
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "3y"], index=2)
    with col3:
        st.markdown("###")
        refresh = st.button("🔄 Refresh Data")

    ticker_display = ticker.replace(".NS", "")
    df = fetch_stock_data(ticker, period=period, force_refresh=refresh)
    df_feat = add_technical_indicators(df)

    price_col, change_col, volume_col, pred_col = st.columns(4)
    current_price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100

    price_col.metric("Current Price", f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    change_col.metric("Day Range", f"₹{df['low'].iloc[-1]:.2f} - ₹{df['high'].iloc[-1]:.2f}")
    volume_col.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")
    volume_sma = df["volume"].tail(20).mean()
    vol_ratio = df["volume"].iloc[-1] / volume_sma
    volume_col.metric("Vol Ratio", f"{vol_ratio:.2f}x")

    if models_exist(ticker):
        pred_dir, pred_prob, conf = make_prediction(ticker, df_feat)
        if pred_dir == 1:
            pred_col.metric("Prediction", "📈 UP", f"{conf:.1f}% confidence")
        else:
            pred_col.metric("Prediction", "📉 DOWN", f"{conf:.1f}% confidence")
    else:
        pred_col.metric("Prediction", "Not trained")

    st.plotly_chart(plot_candlestick(df, ticker_display), use_container_width=True)

    with st.expander("📊 Technical Indicators", expanded=False):
        cols = st.columns(4)
        rsi_val = df_feat["rsi"].iloc[-1]
        rsi_label = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
        cols[0].metric("RSI (14)", f"{rsi_val:.1f}", rsi_label)

        macd_val = df_feat["macd"].iloc[-1]
        macd_signal = df_feat["macd_signal"].iloc[-1]
        macd_status = "Bullish" if macd_val > macd_signal else "Bearish"
        cols[1].metric("MACD", f"{macd_val:.2f}", macd_status)

        bb_width = df_feat["bb_width"].iloc[-1]
        cols[2].metric("Bollinger Width", f"{bb_width:.2f}")

        atr_val = df_feat["atr"].iloc[-1]
        cols[3].metric("ATR (14)", f"{atr_val:.2f}")

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("### 📋 Recent Data")
        display_df = df.tail(10)[["open", "high", "low", "close", "volume"]].copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df.columns = [c.capitalize() for c in display_df.columns]
        st.dataframe(display_df[::-1], use_container_width=True)

    with col_right:
        st.markdown("### 🤖 Model Controls")
        if not models_exist(ticker):
            st.warning("Models not trained for this stock yet!")
            if st.button(f"Train Models for {ticker_display}", type="primary"):
                with st.spinner("Training models... This may take a few minutes."):
                    result = train_for_ticker(ticker, force_retrain=True)
                st.success(f"Training complete! XGBoost: {result['xgb_accuracy']:.2%} | LSTM: {result['lstm_accuracy']:.2%}")
                st.rerun()
        else:
            st.info("✅ Models are trained and ready")
            if st.button("🔄 Retrain Models"):
                with st.spinner("Retraining..."):
                    result = train_for_ticker(ticker, force_retrain=True)
                st.success(f"Retrained! XGBoost: {result['xgb_accuracy']:.2%} | LSTM: {result['lstm_accuracy']:.2%}")
                st.rerun()

    st.markdown("### 📈 Historical Performance")
    df_feat["sma_50"] = df_feat["sma_50"]
    df_feat["sma_200"] = df_feat.get("sma_50", pd.Series(index=df_feat.index))

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="white")))
    fig2.add_trace(go.Scatter(x=df.index, y=df_feat["sma_50"], name="SMA 50", line=dict(color="orange")))
    fig2.add_trace(go.Scatter(x=df.index, y=df_feat["sma_200"], name="SMA 200", line=dict(color="purple")))
    fig2.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)


def make_prediction(ticker: str, df_feat: pd.DataFrame):
    lstm, xgb, scaler, feature_cols = load_models(ticker)
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    latest_data = df_feat[feature_cols].dropna()
    latest_scaled = scaler.transform(latest_data.values[-60:])

    lstm_input = np.expand_dims(latest_scaled, axis=0)
    lstm_pred_scaled = lstm.predict(lstm_input, verbose=0)[0, 0]

    current_close_scaled = latest_scaled[-1, 0]
    lstm_direction = 1 if lstm_pred_scaled > current_close_scaled else 0

    xgb_input = latest_data.iloc[-1:][feature_cols]
    xgb_prob = xgb.predict_proba(xgb_input)[0]
    xgb_direction = int(xgb.predict(xgb_input)[0])

    ensemble_prob = (lstm_direction + xgb_prob[1]) / 2
    ensemble_dir = 1 if ensemble_prob > 0.5 else 0
    confidence = max(ensemble_prob, 1 - ensemble_prob) * 100

    return ensemble_dir, ensemble_prob, confidence


def main():
    prediction_page()


if __name__ == "__main__":
    main()

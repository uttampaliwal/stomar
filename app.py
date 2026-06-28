import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
import traceback

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
    .pred-up {color: #00ff88; font-weight: 700; font-size: 1.2rem;}
    .pred-down {color: #ff4444; font-weight: 700; font-size: 1.2rem;}
    .stMetric {background: #1e1e1e; padding: 1rem; border-radius: 10px; border: 1px solid #333;}
    .feature-bar {margin: 2px 0;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(ttl=3600)
def get_data(ticker: str, period: str, refresh: bool):
    return fetch_stock_data(ticker, period=period, force_refresh=refresh)


def plot_candlestick(df: pd.DataFrame, df_feat: pd.DataFrame, ticker: str):
    df = df.copy()
    df["sma_20"] = df_feat["sma_20"]
    df["sma_50"] = df_feat["sma_50"]
    df["rsi"] = df_feat["rsi"]

    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=(f"{ticker} Price", "RSI (14)", "Volume"),
    )

    fig.add_trace(
        go.Candlestick(
            x=df.index[-90:], open=df["open"][-90:], high=df["high"][-90:],
            low=df["low"][-90:], close=df["close"][-90:], name="Price",
        ), row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index[-90:], y=df["sma_20"][-90:], name="SMA 20", line=dict(color="orange", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index[-90:], y=df["sma_50"][-90:], name="SMA 50", line=dict(color="purple", width=1)),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=df.index[-90:], y=df["rsi"][-90:], name="RSI", line=dict(color="cyan")),
        row=2, col=1,
    )
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    fig.add_trace(
        go.Bar(x=df.index[-90:], y=df["volume"][-90:], name="Volume", marker_color="gray"),
        row=3, col=1,
    )

    fig.update_layout(
        height=650, template="plotly_dark",
        xaxis_rangeslider_visible=False, margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def make_prediction(ticker: str, df_feat: pd.DataFrame):
    import torch
    from src.model import DEVICE

    lstm, xgb, scaler, feature_cols = load_models(ticker)
    feature_cols = [c for c in feature_cols if c in df_feat.columns]

    latest_data = df_feat[feature_cols].dropna()
    if len(latest_data) < 60:
        return None, None, None, None

    latest_scaled = scaler.transform(latest_data.values[-60:])
    lstm_input = torch.tensor(latest_scaled, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    lstm.eval()
    with torch.no_grad():
        lstm_pred_scaled = lstm(lstm_input).item()

    current_close_scaled = latest_scaled[-1, 0]
    lstm_direction = 1 if lstm_pred_scaled > current_close_scaled else 0

    xgb_input = latest_data.iloc[-1:][feature_cols]
    xgb_prob = xgb.predict_proba(xgb_input)[0]
    xgb_direction = int(xgb.predict(xgb_input)[0])

    ensemble_prob = (lstm_direction + xgb_prob[1]) / 2
    ensemble_dir = 1 if ensemble_prob > 0.5 else 0
    confidence = max(ensemble_prob, 1 - ensemble_prob) * 100

    return ensemble_dir, confidence, xgb_prob[1], lstm_direction


def show_feature_importance(xgb_model, feature_cols):
    importance = xgb_model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importance})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(15)

    fig = go.Figure(go.Bar(
        x=feat_df["importance"], y=feat_df["feature"],
        orientation="h", marker_color="limegreen",
    ))
    fig.update_layout(
        template="plotly_dark", height=350,
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="Importance", yaxis_title=None,
    )
    return fig


def prediction_page():
    st.markdown('<p class="main-header">StoMar 📈</p>', unsafe_allow_html=True)
    st.markdown("### Indian Stock Market Predictor (NSE)")

    col1, col2, col3, col4 = st.columns([2, 1, 1, 0.8])
    with col1:
        ticker = st.selectbox("Select Stock", NSE_STOCKS, index=0)
    with col2:
        period = st.selectbox("Data Period", ["6mo", "1y", "2y", "5y"], index=3)
    with col3:
        st.markdown("###")
        refresh = st.button("🔄 Refresh Data")
    with col4:
        st.markdown("###")
        train_btn = st.button("⚡ Train Model")

    market_status = get_market_status()
    status_icon = "🟢" if market_status == "Open" else ("🟡" if "Closed" in market_status else "🔴")
    st.markdown(f"**Market:** {status_icon} {market_status}")

    try:
        df = get_data(ticker, period, refresh)
    except Exception as e:
        st.error(f"Failed to fetch data: {e}")
        st.stop()

    ticker_display = ticker.replace(".NS", "")
    df_feat = add_technical_indicators(df)

    # Top metrics row
    current_price = df["close"].iloc[-1]
    prev_close = df["close"].iloc[-2]
    change = current_price - prev_close
    change_pct = (change / prev_close) * 100

    price_col, range_col, vol_col, pred_col = st.columns(4)
    price_col.metric("Current Price", f"₹{current_price:.2f}", f"{change:+.2f} ({change_pct:+.2f}%)")
    range_col.metric("Day Range", f"₹{df['low'].iloc[-1]:.2f} – ₹{df['high'].iloc[-1]:.2f}")
    vol_col.metric("Volume", f"{df['volume'].iloc[-1]:,.0f}")

    # Prediction display
    if train_btn:
        with st.spinner(f"Training models for {ticker_display}... (~2 min)"):
            try:
                result = train_for_ticker(ticker, force_retrain=True)
                st.success(
                    f"✅ Training done!  "
                    f"XGBoost: {result['xgb_accuracy']:.1%}  |  "
                    f"LSTM: {result['lstm_accuracy']:.1%}  "
                    f"({result.get('lstm_correct',0)}/{result.get('lstm_total',0)} correct)"
                )
                st.cache_data.clear()
            except Exception as e:
                st.error(f"Training failed: {e}")

    if models_exist(ticker):
        pred_dir, conf, xgb_prob, lstm_dir = make_prediction(ticker, df_feat)
        if pred_dir is not None:
            direction_text = "📈 UP" if pred_dir == 1 else "📉 DOWN"
            direction_class = "pred-up" if pred_dir == 1 else "pred-down"
            pred_col.markdown(
                f"**Prediction**<br>"
                f'<span class="{direction_class}">{direction_text}</span><br>'
                f"<small>{conf:.1f}% confidence</small>",
                unsafe_allow_html=True,
            )
        else:
            pred_col.metric("Prediction", "⏳ Need 60d")
    else:
        pred_col.metric("Prediction", "Not trained")

    # Chart
    st.plotly_chart(plot_candlestick(df, df_feat, ticker_display), use_container_width=True)

    # Technical indicators expander
    with st.expander("📊 Technical Indicators", expanded=False):
        cols = st.columns(5)
        rsi_val = df_feat["rsi"].iloc[-1]
        rsi_label = "Overbought ⚠️" if rsi_val > 70 else "Oversold ⚠️" if rsi_val < 30 else "Neutral"
        cols[0].metric("RSI (14)", f"{rsi_val:.1f}", rsi_label)

        macd_val = df_feat["macd"].iloc[-1]
        macd_sig = df_feat["macd_signal"].iloc[-1]
        macd_status = "Bullish ✅" if macd_val > macd_sig else "Bearish"
        cols[1].metric("MACD", f"{macd_val:.2f}", macd_status)

        bb_w = df_feat["bb_width"].iloc[-1]
        cols[2].metric("Bollinger Width", f"{bb_w:.2f}")

        atr_v = df_feat["atr"].iloc[-1]
        cols[3].metric("ATR (14)", f"{atr_v:.2f}")

        obv_v = df_feat["obv"].iloc[-1]
        cols[4].metric("OBV", f"{obv_v:,.0f}")

    # Bottom: two columns
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### 📋 Recent Data")
        display_df = df.tail(8)[["open", "high", "low", "close", "volume"]].copy()
        display_df.index = display_df.index.strftime("%Y-%m-%d")
        display_df.columns = [c.capitalize() for c in display_df.columns]
        st.dataframe(display_df.iloc[::-1], use_container_width=True)

        # Feature importance
        if models_exist(ticker):
            try:
                _, xgb, _, _ = load_models(ticker)
                if hasattr(xgb, "feature_importances_"):
                    st.markdown("### 🔑 Feature Importance (XGBoost)")
                    feat_cols = [c for c in FEATURE_COLS if c in df_feat.columns]
                    fig_fi = show_feature_importance(xgb, feat_cols)
                    st.plotly_chart(fig_fi, use_container_width=True)
            except Exception:
                pass

    with col_right:
        st.markdown("### 🤖 Model Controls")
        if not models_exist(ticker):
            st.warning("Models not trained for this stock")
            if st.button(f"Train {ticker_display}", type="primary", use_container_width=True):
                with st.spinner("Training... (~2 min)"):
                    result = train_for_ticker(ticker, force_retrain=True)
                st.success(f"XGB: {result['xgb_accuracy']:.1%} | LSTM: {result['lstm_accuracy']:.1%}")
                st.cache_data.clear()
                st.rerun()
        else:
            _, xgb, _, _ = load_models(ticker)
            lstm, _, _, _ = load_models(ticker)
            st.info("✅ Models trained and ready")
            if st.button("🔄 Retrain", use_container_width=True):
                with st.spinner("Retraining..."):
                    result = train_for_ticker(ticker, force_retrain=True)
                st.success(f"XGB: {result['xgb_accuracy']:.1%} | LSTM: {result['lstm_accuracy']:.1%}")
                st.cache_data.clear()
                st.rerun()

        # Backtest summary
        if models_exist(ticker):
            st.markdown("### 📈 Backtest Summary (last 20% of data)")
            try:
                from sklearn.metrics import accuracy_score
                lstm, xgb, scaler, feature_cols = load_models(ticker)
                feature_cols = [c for c in feature_cols if c in df_feat.columns]
                bt_data = df_feat[feature_cols].dropna()
                if len(bt_data) > 100:
                    split = int(len(bt_data) * 0.8)
                    bt_test = bt_data.iloc[split:]
                    bt_scaled = scaler.transform(bt_test.values)
                    correct_lstm = 0
                    total = len(bt_scaled) - 60
                    for i in range(60, len(bt_scaled)):
                        inp = torch.tensor(bt_scaled[i-60:i], dtype=torch.float32).unsqueeze(0)
                        lstm.eval()
                        with torch.no_grad():
                            p = lstm(inp).item()
                        actual = bt_scaled[i, 0]
                        prev = bt_scaled[i-1, 0]
                        pred_dir = 1 if p > prev else 0
                        act_dir = 1 if actual > prev else 0
                        if pred_dir == act_dir:
                            correct_lstm += 1
                    bt_acc = correct_lstm / total if total > 0 else 0

                    xgb_test = bt_test[feature_cols]
                    xgb_actual = df_feat["target_direction"].iloc[split:split+len(xgb_test)]
                    xgb_pred = xgb.predict(xgb_test)
                    xgb_valid = min(len(xgb_pred), len(xgb_actual))
                    xgb_acc = accuracy_score(xgb_actual[:xgb_valid], xgb_pred[:xgb_valid])

                    col_a, col_b = st.columns(2)
                    col_a.metric("LSTM Accuracy", f"{bt_acc:.1%}")
                    col_b.metric("XGBoost Accuracy", f"{xgb_acc:.1%}")
            except Exception as e:
                st.caption(f"Backtest unavailable: {e[:50]}...")

    # Long-term chart
    st.markdown("### 📈 Long-Term Trend")
    sma_200 = df["close"].rolling(200).mean()
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=df.index, y=df["close"], name="Close", line=dict(color="white")))
    fig2.add_trace(go.Scatter(x=df.index, y=df_feat["sma_50"], name="SMA 50", line=dict(color="orange", width=1)))
    fig2.add_trace(go.Scatter(x=df.index, y=sma_200, name="SMA 200", line=dict(color="purple", width=1)))
    fig2.update_layout(template="plotly_dark", height=400, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig2, use_container_width=True)

    # Footer
    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This is for educational purposes only. "
        "Stock predictions are inherently uncertain. Do not trade based solely on these predictions."
    )


FEATURE_COLS = [
    "close", "volume", "sma_10", "sma_20", "sma_50", "ema_12", "ema_26",
    "rsi", "macd", "macd_signal", "bb_width", "atr", "obv", "volume_ratio",
    "high_low_pct", "close_open_pct", "close_position",
    "returns_1d", "returns_2d", "returns_3d", "returns_5d", "returns_10d", "returns_20d",
    "volatility_5d", "volatility_10d", "volatility_20d",
    "return_lag_1", "return_lag_2", "return_lag_3", "return_lag_5",
    "day_of_week", "month", "quarter", "day_of_month",
]


def main():
    prediction_page()


if __name__ == "__main__":
    main()

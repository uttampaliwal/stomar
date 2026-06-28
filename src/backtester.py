import pandas as pd
import numpy as np
from src.portfolio import Portfolio


def walk_forward_split(df, train_years=3, test_years=1, step_months=6):
    train_days = train_years * 252
    test_days = test_years * 252
    step_days = step_months * 21

    splits = []
    start = 0
    while start + train_days + test_days <= len(df):
        train_end = start + train_days
        test_end = train_end + test_days
        splits.append({
            "train": df.iloc[start:train_end].index,
            "test": df.iloc[train_end:test_end].index,
        })
        start += step_days

    return splits


def compute_metrics(equity_curve, trades, risk_free_rate=0.065):
    if len(equity_curve) < 2:
        return {}

    eq = pd.Series(equity_curve["equity"].values, index=pd.to_datetime(equity_curve["date"]))
    returns = eq.pct_change().dropna()

    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0]
    downside_vol = downside.std() * np.sqrt(252) if len(downside) > 0 else 0.001
    sortino = (ann_return - risk_free_rate) / downside_vol

    cummax = eq.cummax()
    drawdown = (eq - cummax) / cummax
    max_dd = drawdown.min()
    max_dd_duration = 0
    current_dd = 0
    for dd in drawdown:
        if dd < 0:
            current_dd += 1
            max_dd_duration = max(max_dd_duration, current_dd)
        else:
            current_dd = 0

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    var_95 = returns.quantile(0.05) if len(returns) > 20 else 0
    cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95

    win_trades = [t for t in trades if t.get("pnl", 0) > 0]
    lose_trades = [t for t in trades if t.get("pnl", 0) <= 0]
    total_trades = len(trades)
    win_rate = len(win_trades) / max(total_trades, 1)

    avg_win = np.mean([t["pnl"] for t in win_trades]) if win_trades else 0
    avg_loss = abs(np.mean([t["pnl"] for t in lose_trades])) if lose_trades else 0.001
    profit_factor = (avg_win * len(win_trades)) / max(avg_loss * len(lose_trades), 0.001)

    return {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_volatility": ann_vol,
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio": round(sortino, 3),
        "calmar_ratio": round(calmar, 3),
        "max_drawdown": round(max_dd * 100, 2),
        "max_dd_duration_days": max_dd_duration,
        "var_95": round(var_95 * 100, 3),
        "cvar_95": round(cvar_95 * 100, 3),
        "total_trades": total_trades,
        "win_rate": round(win_rate * 100, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "n_years": round(n_years, 1),
    }


def run_walk_forward_backtest(
    ticker,
    df_feat,
    feature_cols,
    train_years=3,
    test_years=1,
    step_months=6,
    initial_capital=100000,
    position_pct=0.25,
    stop_loss_pct=0.05,
    take_profit_pct=0.15,
    max_positions=5,
    brokerage=0.0003,
    slippage=0.001,
):
    from src.model import build_lstm, build_gru, build_transformer, build_xgb_model, DEVICE
    from src.trainer import _train_one_model, SEQ_LENGTH, EPOCHS, BATCH_SIZE, LEARNING_RATE
    from src.features import add_technical_indicators
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import accuracy_score
    from torch.utils.data import DataLoader, TensorDataset
    import torch

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    df_clean = df_feat[feature_cols + ["close"]].dropna()
    df_clean = df_clean[~df_clean.index.duplicated(keep="first")]
    dates = df_clean.index.tolist()
    values = df_clean.values
    n = len(df_clean)
    feat_idx = {c: i for i, c in enumerate(feature_cols)}
    close_idx = len(feature_cols)

    splits = walk_forward_split(df_clean, train_years, test_years, step_months)
    if not splits:
        return None, None, []

    all_test_results = []
    portfolio = Portfolio(initial_capital)
    equity_points = []

    print(f"Walk-forward: {len(splits)} windows, {train_years}y train / {test_years}y test")

    for wi, split in enumerate(splits):
        train_dates_in_split = split["train"]
        test_dates_in_split = split["test"]

        train_positions = [dates.index(d) for d in train_dates_in_split]
        test_positions = [dates.index(d) for d in test_dates_in_split]

        train_values = values[train_positions]
        test_values = values[test_positions]
        train_dates = [dates[p] for p in train_positions]
        test_dates = [dates[p] for p in test_positions]

        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train_values[:, :close_idx])

        X_train, y_train = [], []
        for i in range(SEQ_LENGTH, len(train_scaled)):
            X_train.append(train_scaled[i - SEQ_LENGTH:i])
            y_train.append(train_scaled[i, 0])
        X_train = np.array(X_train, dtype=np.float32)
        y_train = np.array(y_train, dtype=np.float32)

        if len(X_train) < 10:
            continue

        split_point = int(len(X_train) * 0.85)
        X_tr = torch.tensor(X_train[:split_point]).to(DEVICE)
        y_tr = torch.tensor(y_train[:split_point]).to(DEVICE)
        X_val = torch.tensor(X_train[split_point:]).to(DEVICE)
        y_val = torch.tensor(y_train[split_point:]).to(DEVICE)

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=True)
        input_dim = X_train.shape[2]

        lstm = build_lstm(input_dim)
        gru = build_gru(input_dim)
        transformer = build_transformer(input_dim)
        xgb = build_xgb_model()

        lstm = _train_one_model(lstm, loader, X_val, y_val, f"W{wi+1}-LSTM", epochs=30)
        gru = _train_one_model(gru, loader, X_val, y_val, f"W{wi+1}-GRU", epochs=30)
        transformer = _train_one_model(transformer, loader, X_val, y_val, f"W{wi+1}-TF", epochs=30)

        xgb_feat = train_values[SEQ_LENGTH:, :close_idx]
        xgb_close_vals = train_values[SEQ_LENGTH:, close_idx]
        xgb_target = np.zeros(len(xgb_close_vals), dtype=int)
        for j in range(len(xgb_close_vals) - 1):
            if xgb_close_vals[j + 1] > xgb_close_vals[j]:
                xgb_target[j] = 1
        valid_mask = ~np.isnan(xgb_feat).any(axis=1)
        xgb.fit(xgb_feat[valid_mask], xgb_target[valid_mask], verbose=False)

        test_scaled = scaler.transform(test_values[:, :close_idx])

        for i in range(SEQ_LENGTH, len(test_scaled)):
            date_val = test_dates[i]
            date = str(date_val.date()) if hasattr(date_val, 'date') else str(date_val)[:10]
            inp = torch.tensor(test_scaled[i-SEQ_LENGTH:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

            prev_close_raw = test_values[i-1, close_idx]
            curr_close_raw = test_values[i, close_idx]
            actual_dir = 1 if curr_close_raw > prev_close_raw else 0

            lstm.eval()
            gru.eval()
            transformer.eval()
            with torch.no_grad():
                p_l = lstm(inp).item()
                p_g = gru(inp).item()
                p_t = transformer(inp).item()

            prev_scaled = test_scaled[i-1, 0]
            d_l = 1 if p_l > prev_scaled else 0
            d_g = 1 if p_g > prev_scaled else 0
            d_t = 1 if p_t > prev_scaled else 0

            xgb_inp = test_values[[i-1], :close_idx]
            xgb_p = xgb.predict_proba(xgb_inp)[0][1]

            dl_votes = d_l + d_g + d_t
            dl_dir = 1 if dl_votes >= 2 else 0
            ensemble_prob = (dl_dir * 0.4 + xgb_p * 0.6)
            final_dir = 1 if ensemble_prob > 0.5 else 0
            confidence = abs(ensemble_prob - 0.5) * 2

            all_test_results.append({
                "date": date,
                "predicted": final_dir,
                "actual": actual_dir,
                "confidence": confidence,
                "price": curr_close_raw,
                "ensemble_prob": ensemble_prob,
                "lstm": d_l, "gru": d_g, "transformer": d_t, "xgb_prob": xgb_p,
            })

            equity_points.append({"date": date, "equity": portfolio.portfolio_value()})

    if not all_test_results:
        return None, None, []

    pred_arr = np.array([r["predicted"] for r in all_test_results])
    act_arr = np.array([r["actual"] for r in all_test_results])
    ensemble_acc = accuracy_score(act_arr, pred_arr)

    returns = []
    for i in range(1, len(all_test_results)):
        r = all_test_results[i]
        prev_r = all_test_results[i-1]
        if prev_r["predicted"] == 1 and r["actual"] == 1:
            returns.append(0.01)
        elif prev_r["predicted"] == 1 and r["actual"] == 0:
            returns.append(-0.01)
        else:
            returns.append(0.0)

    metrics = {
        "ensemble_accuracy": ensemble_acc,
        "total_test_days": len(all_test_results),
        "n_windows": len(splits),
    }

    if returns:
        ret_arr = np.array(returns)
        metrics["simulated_annual_return"] = float(ret_arr.mean() * 252)
        metrics["simulated_sharpe"] = float(ret_arr.mean() / max(ret_arr.std(), 0.001) * np.sqrt(252))
        metrics["simulated_win_rate"] = float((ret_arr > 0).mean() * 100)

    if equity_points:
        eq_df = pd.DataFrame(equity_points)
        eq_metrics = compute_metrics(eq_df, [])
        metrics.update(eq_metrics)

    return metrics, portfolio, all_test_results


def run_simple_backtest(df_feat, signals, initial_capital=100000, brokerage=0.0003, slippage=0.001):
    portfolio = Portfolio(initial_capital)
    in_position = {}
    tickers = list(signals.keys())

    for date_key in sorted(set(
        d for sigs in signals.values() for d in sigs.keys()
    )):
        date = pd.Timestamp(date_key)
        for ticker in tickers:
            if date_key not in signals[ticker]:
                continue
            signal = signals[ticker][date_key]
            price = signal.get("price", 0)
            if price == 0:
                continue

            if signal["direction"] == 1 and not in_position.get(ticker):
                exec_price = price * (1 + slippage)
                qty = int(portfolio.cash * 0.25 / exec_price)
                if qty > 0:
                    portfolio.buy(ticker, exec_price, qty, date, brokerage)
                    in_position[ticker] = {"qty": qty, "entry": exec_price, "date": date}

            elif signal["direction"] == 0 and in_position.get(ticker):
                exec_price = price * (1 - slippage)
                pos = in_position[ticker]
                portfolio.sell(ticker, exec_price, pos["qty"], date, brokerage)
                in_position[ticker] = None

    for ticker, pos in in_position.items():
        if pos and pos.get("qty", 0) > 0:
            last_price = signals[ticker].get(
                max(signals[ticker].keys()), {}
            ).get("price", 0)
            if last_price > 0:
                portfolio.sell(ticker, last_price * (1 - slippage), pos["qty"], pd.Timestamp.now(), brokerage)

    stats = portfolio.get_stats()
    if stats and stats.get("total_trades", 0) > 0:
        equity_points = portfolio.equity_curve
        if equity_points:
            eq_df = pd.DataFrame(equity_points)
            all_trades = portfolio.trades
            detailed = compute_metrics(eq_df, all_trades)
            stats.update(detailed)

    return stats, portfolio


def generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feature_cols):
    import torch
    from src.model import DEVICE

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols].dropna()
    scaled = scaler.transform(data.values)

    signals = {}
    for i in range(60, len(scaled)):
        date = str(data.index[i].date())
        inp = torch.tensor(scaled[i-60:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

        prev_close = scaled[i-1, 0]

        lstm.eval()
        gru.eval()
        transformer.eval()
        with torch.no_grad():
            p_l = lstm(inp).item()
            p_g = gru(inp).item()
            p_t = transformer(inp).item()

        d_l = 1 if p_l > prev_close else 0
        d_g = 1 if p_g > prev_close else 0
        d_t = 1 if p_t > prev_close else 0

        xgb_inp = data.iloc[[i-1]][feature_cols]
        xgb_p = xgb.predict_proba(xgb_inp)[0][1]

        ens = d_l + d_g + d_t
        dl_dir = 1 if ens >= 2 else 0
        final = 1 if (dl_dir * 0.4 + xgb_p * 0.6) > 0.5 else 0
        conf = abs((dl_dir * 0.4 + xgb_p * 0.6) - 0.5) * 2

        signals[date] = {
            "direction": final,
            "confidence": conf,
            "price": float(data.loc[data.index[i], "close"]),
            "actual": 1 if scaled[i, 0] > prev_close else 0,
        }
    return {ticker: signals}

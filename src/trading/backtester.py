import pandas as pd
import numpy as np
from src.trading.portfolio import Portfolio
from src.core.constants import BROKERAGE_RATE, SLIPPAGE_RATE, RISK_FREE_RATE
from src.core.logging_config import get_logger

logger = get_logger("backtester")


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


def compute_metrics(equity_curve, trades, risk_free_rate=RISK_FREE_RATE):
    if len(equity_curve) < 2:
        return {}

    eq = pd.Series(equity_curve["equity"].values, index=pd.to_datetime(equity_curve["date"]))
    returns = eq.pct_change().dropna()

    if eq.iloc[0] == 0:
        return {}

    total_return = (eq.iloc[-1] / eq.iloc[0]) - 1
    n_years = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    ann_return = (1 + total_return) ** (1 / n_years) - 1
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_return - risk_free_rate) / ann_vol if ann_vol > 0 else 0

    daily_mar = risk_free_rate / 252
    downside_diff = np.minimum(returns.values - daily_mar, 0)
    downside_vol = np.sqrt(np.mean(downside_diff ** 2)) * np.sqrt(252)
    sortino = (ann_return - risk_free_rate) / downside_vol if downside_vol > 0 else 0

    cummax = eq.cummax()
    cummax_safe = cummax.replace(0, np.nan)
    drawdown = (eq - cummax) / cummax_safe
    drawdown = drawdown.fillna(0)
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

    sell_trades = [t for t in trades if t.get("pnl") is not None]
    win_trades = [t for t in sell_trades if t.get("pnl", 0) > 0]
    lose_trades = [t for t in sell_trades if t.get("pnl", 0) < 0]
    total_trades = len(sell_trades)
    win_rate = len(win_trades) / max(total_trades, 1)

    avg_win = np.mean([t.get("pnl", 0) for t in win_trades]) if win_trades else 0
    avg_loss = abs(np.mean([t.get("pnl", 0) for t in lose_trades])) if lose_trades else 0
    total_win_pnl = avg_win * len(win_trades) if win_trades else 0
    total_loss_pnl = avg_loss * len(lose_trades) if lose_trades else 0
    profit_factor = total_win_pnl / max(total_loss_pnl, 0.001)

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
    brokerage=BROKERAGE_RATE,
    slippage=SLIPPAGE_RATE,
):
    from src.models.model import build_lstm, build_gru, build_transformer, build_xgb_model, DEVICE
    from src.models.trainer import _train_one_model, SEQ_LENGTH, BATCH_SIZE
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
    in_position = False

    logger.info("walk_forward_start windows=%d train_years=%d test_years=%d", len(splits), train_years, test_years)

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

        loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=BATCH_SIZE, shuffle=False)
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
            date_str = str(date_val.date()) if hasattr(date_val, 'date') else str(date_val)[:10]
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
                "date": date_str,
                "predicted": final_dir,
                "actual": actual_dir,
                "confidence": confidence,
                "price": curr_close_raw,
                "ensemble_prob": ensemble_prob,
                "lstm": d_l, "gru": d_g, "transformer": d_t, "xgb_prob": xgb_p,
            })

            exec_price = curr_close_raw * (1 + slippage) if final_dir == 1 else curr_close_raw * (1 - slippage)

            if final_dir == 1 and not in_position:
                qty = int(portfolio.cash * position_pct / exec_price) if exec_price > 0 else 0
                if qty > 0:
                    portfolio.buy(ticker, exec_price, qty, date_val)
                    in_position = True
            elif final_dir == 0 and in_position:
                ticker_key = list(portfolio.holdings.keys())[0] if portfolio.holdings else ticker
                if ticker_key in portfolio.holdings:
                    held_qty = portfolio.holdings[ticker_key][0]
                    portfolio.sell(ticker_key, exec_price, held_qty, date_val)
                    in_position = False

            equity_points.append({"date": date_val, "equity": portfolio.portfolio_value({ticker: curr_close_raw})})

    if in_position and portfolio.holdings:
        last_price = all_test_results[-1]["price"] if all_test_results else 0
        if last_price > 0:
            ticker_key = list(portfolio.holdings.keys())[0]
            held_qty = portfolio.holdings[ticker_key][0]
            portfolio.sell(ticker_key, last_price * (1 - slippage), held_qty,
                           pd.Timestamp(all_test_results[-1]["date"]))
            in_position = False

    if not all_test_results:
        return None, None, []

    pred_arr = np.array([r["predicted"] for r in all_test_results])
    act_arr = np.array([r["actual"] for r in all_test_results])
    ensemble_acc = accuracy_score(act_arr, pred_arr)

    metrics = {
        "ensemble_accuracy": ensemble_acc,
        "total_test_days": len(all_test_results),
        "n_windows": len(splits),
    }

    if equity_points:
        eq_df = pd.DataFrame(equity_points)
        eq_metrics = compute_metrics(eq_df, portfolio.trades)
        metrics.update(eq_metrics)

    return metrics, portfolio, all_test_results


def run_simple_backtest(df_feat, signals, initial_capital=100000, brokerage=BROKERAGE_RATE, slippage=SLIPPAGE_RATE):
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
                qty = int(portfolio.cash * 0.25 / exec_price) if exec_price > 0 else 0
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


def generate_model_signals(ticker, df_feat, lstm, gru, transformer, xgb, scaler, feature_cols, seq_length=60):
    import torch
    from src.models.model import DEVICE

    feature_cols = [c for c in feature_cols if c in df_feat.columns]
    data = df_feat[feature_cols].dropna()
    scaled = scaler.transform(data.values)

    signals = {}
    for i in range(seq_length, len(scaled)):
        date = str(data.index[i].date())
        inp = torch.tensor(scaled[i-seq_length:i], dtype=torch.float32).unsqueeze(0).to(DEVICE)

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


def monte_carlo_backtest(equity_curve, trades, n_simulations=1000, seed=42):
    """Run Monte Carlo simulation by shuffling trade outcomes.

    Tests robustness: if the strategy's edge is real, performance should
    remain positive across most random orderings of trades.

    Args:
        equity_curve: DataFrame with 'date' and 'equity' columns
        trades: List of trade dicts with 'pnl' field
        n_simulations: Number of random shuffles
        seed: Random seed for reproducibility

    Returns:
        Dict with confidence intervals on key metrics
    """
    if not trades or len(trades) < 5:
        return {"error": "Need at least 5 trades for Monte Carlo"}

    rng = np.random.RandomState(seed)
    pnls = np.array([t.get("pnl", 0) for t in trades])

    simulated_returns = []
    simulated_sharpes = []
    simulated_max_dds = []
    simulated_final_equity = []

    initial_capital = equity_curve["equity"].iloc[0] if len(equity_curve) > 0 else 100000

    for _ in range(n_simulations):
        shuffled_pnls = rng.permutation(pnls)
        equity = [initial_capital]
        for pnl in shuffled_pnls:
            equity.append(equity[-1] + pnl)

        eq_arr = np.array(equity)
        rets = np.diff(eq_arr) / eq_arr[:-1]
        rets = rets[np.isfinite(rets)]

        if len(rets) > 1:
            ann_ret = float(np.mean(rets) * 252)
            ann_vol = float(np.std(rets) * np.sqrt(252))
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
            cummax = np.maximum.accumulate(eq_arr)
            dd = (eq_arr - cummax) / np.maximum(cummax, 1)
            max_dd = float(dd.min())

            simulated_returns.append(ann_ret)
            simulated_sharpes.append(sharpe)
            simulated_max_dds.append(max_dd)
            simulated_final_equity.append(eq_arr[-1])

    if not simulated_returns:
        return {"error": "Simulation produced no valid results"}

    ret_arr = np.array(simulated_returns)
    sharpe_arr = np.array(simulated_sharpes)
    dd_arr = np.array(simulated_max_dds)
    eq_arr = np.array(simulated_final_equity)

    return {
        "n_simulations": n_simulations,
        "n_trades": len(trades),
        "return_ci_5": float(np.percentile(ret_arr, 5)),
        "return_ci_50": float(np.percentile(ret_arr, 50)),
        "return_ci_95": float(np.percentile(ret_arr, 95)),
        "return_mean": float(ret_arr.mean()),
        "sharpe_ci_5": float(np.percentile(sharpe_arr, 5)),
        "sharpe_ci_50": float(np.percentile(sharpe_arr, 50)),
        "sharpe_ci_95": float(np.percentile(sharpe_arr, 95)),
        "max_dd_ci_5": float(np.percentile(dd_arr, 5)),
        "max_dd_ci_50": float(np.percentile(dd_arr, 50)),
        "max_dd_ci_95": float(np.percentile(dd_arr, 95)),
        "prob_profit": float((eq_arr > initial_capital).mean()),
        "worst_case_5pct": float(np.percentile(eq_arr, 5)),
        "median_outcome": float(np.percentile(eq_arr, 50)),
        "best_case_95pct": float(np.percentile(eq_arr, 95)),
    }

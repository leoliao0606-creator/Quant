from __future__ import annotations


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'. Install requirements.txt first.") from exc
    return np


def _load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'pandas'. Install requirements.txt first.") from exc
    return pd


def _infer_periods_per_year(timestamps):
    pd = _load_pandas()
    deltas = pd.Series(timestamps).sort_values().diff().dropna()
    if deltas.empty:
        return 252

    median_minutes = deltas.median().total_seconds() / 60.0
    if median_minutes <= 0.0:
        return 252

    bars_per_day = max(int(round(390.0 / median_minutes)), 1)
    return bars_per_day * 252


def _max_drawdown(equity_curve):
    running_peak = equity_curve.cummax()
    drawdown = equity_curve / running_peak - 1.0
    return float(drawdown.min())


def simulate_probability_strategy(
    prediction_rows,
    entry_probability: float,
    exit_probability: float,
    transaction_cost_bps: float = 0.0,
    max_active_positions: int | None = None,
):
    pd = _load_pandas()
    np = _load_numpy()

    required_columns = {"timestamp", "symbol", "probability_up", "next_bar_return"}
    missing_columns = required_columns.difference(prediction_rows.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for backtest: {sorted(missing_columns)}")

    rows = prediction_rows.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    if max_active_positions is not None and max_active_positions <= 0:
        max_active_positions = None

    position_by_symbol = {symbol: 0 for symbol in rows["symbol"].unique().tolist()}
    symbol_records = []
    trade_count = 0

    for timestamp, timestamp_rows in rows.groupby("timestamp", sort=True):
        ranked_rows = timestamp_rows.sort_values("probability_up", ascending=False)
        changed_symbols = set()

        for row in ranked_rows.itertuples(index=False):
            if position_by_symbol[row.symbol] == 1 and row.probability_up <= exit_probability:
                position_by_symbol[row.symbol] = 0
                changed_symbols.add(row.symbol)
                trade_count += 1

        current_active_positions = sum(position_by_symbol.values())
        available_slots = None
        if max_active_positions is not None:
            available_slots = max(max_active_positions - current_active_positions, 0)

        entry_candidates = []
        for row in ranked_rows.itertuples(index=False):
            if position_by_symbol[row.symbol] == 0 and row.probability_up >= entry_probability:
                entry_candidates.append(row)

        if available_slots is None:
            allowed_entries = entry_candidates
        else:
            allowed_entries = entry_candidates[:available_slots]

        for row in allowed_entries:
            if position_by_symbol[row.symbol] == 0:
                position_by_symbol[row.symbol] = 1
                changed_symbols.add(row.symbol)
                trade_count += 1

        for row in ranked_rows.itertuples(index=False):
            transaction_cost = 0.0
            if row.symbol in changed_symbols:
                transaction_cost = transaction_cost_bps / 10000.0

            strategy_return = position_by_symbol[row.symbol] * float(row.next_bar_return) - transaction_cost
            symbol_records.append(
                {
                    "timestamp": timestamp,
                    "symbol": row.symbol,
                    "position": position_by_symbol[row.symbol],
                    "strategy_return": strategy_return,
                }
            )

    if not symbol_records:
        empty = pd.DataFrame(columns=["timestamp", "portfolio_return", "equity_curve"])
        return {
            "entry_probability": float(entry_probability),
            "exit_probability": float(exit_probability),
            "max_active_positions": max_active_positions,
            "trade_count": 0,
            "exposure": 0.0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "equity_curve": empty,
        }

    symbol_frame = pd.DataFrame(symbol_records)
    portfolio = (
        symbol_frame.groupby("timestamp", sort=True)
        .agg(
            portfolio_return=("strategy_return", "mean"),
            active_positions=("position", "sum"),
        )
        .reset_index()
    )
    portfolio["equity_curve"] = (1.0 + portfolio["portfolio_return"]).cumprod()

    periods_per_year = _infer_periods_per_year(portfolio["timestamp"])
    mean_return = float(portfolio["portfolio_return"].mean())
    volatility = float(portfolio["portfolio_return"].std(ddof=0))
    annualized_return = float((portfolio["equity_curve"].iloc[-1] ** (periods_per_year / max(len(portfolio), 1))) - 1.0)
    annualized_volatility = float(volatility * np.sqrt(periods_per_year))
    sharpe = 0.0
    if volatility > 0.0:
        sharpe = float(mean_return / volatility * np.sqrt(periods_per_year))

    exposure = float(symbol_frame["position"].mean()) if not symbol_frame.empty else 0.0
    return {
        "entry_probability": float(entry_probability),
        "exit_probability": float(exit_probability),
        "max_active_positions": max_active_positions,
        "trade_count": int(trade_count),
        "exposure": exposure,
        "total_return": float(portfolio["equity_curve"].iloc[-1] - 1.0),
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(portfolio["equity_curve"]),
        "equity_curve": portfolio[["timestamp", "portfolio_return", "equity_curve"]].copy(),
    }


def select_probability_thresholds(
    validation_rows,
    transaction_cost_bps: float,
    threshold_hysteresis: float,
    max_active_positions: int | None,
):
    np = _load_numpy()

    candidate_entries = np.arange(0.45, 0.71, 0.02)
    best_choice = None

    for entry_probability in candidate_entries:
        exit_probability = max(entry_probability - threshold_hysteresis, 0.05)
        result = simulate_probability_strategy(
            prediction_rows=validation_rows,
            entry_probability=float(entry_probability),
            exit_probability=float(exit_probability),
            transaction_cost_bps=transaction_cost_bps,
            max_active_positions=max_active_positions,
        )
        if result["trade_count"] < 4:
            continue

        score = (result["sharpe"], result["total_return"], -abs(result["exposure"] - 0.35))
        if best_choice is None or score > best_choice["score"]:
            best_choice = {
                "score": score,
                "entry_probability": float(entry_probability),
                "exit_probability": float(exit_probability),
                "validation_backtest": result,
            }

    if best_choice is not None:
        return best_choice

    fallback_entry = 0.55
    fallback_exit = max(fallback_entry - threshold_hysteresis, 0.05)
    result = simulate_probability_strategy(
        prediction_rows=validation_rows,
        entry_probability=fallback_entry,
        exit_probability=fallback_exit,
        transaction_cost_bps=transaction_cost_bps,
        max_active_positions=max_active_positions,
    )
    return {
        "score": (result["sharpe"], result["total_return"], -abs(result["exposure"] - 0.35)),
        "entry_probability": fallback_entry,
        "exit_probability": fallback_exit,
        "validation_backtest": result,
    }

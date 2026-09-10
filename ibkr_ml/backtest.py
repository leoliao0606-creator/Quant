from __future__ import annotations

from dataclasses import dataclass

from .config import ModelConfig, RiskConfig
from .strategy import generate_trade_decision


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


@dataclass(slots=True)
class _SimulatedPosition:
    quantity: int = 0
    average_cost: float = 0.0
    bars_held: int = 0


class _PortfolioSimulator:
    """Replay prediction rows through the decision function the live loop uses.

    The previous simulator carried its own rule - "probability above the entry
    threshold means hold one unit" - which shared nothing with
    strategy.generate_trade_decision beyond the two thresholds. It had no stop
    loss, no take profit, no daily loss limit, no daily trade cap, and it never
    closed the book at the end of a session. Since execution.py reads Sharpe
    and drawdown out of this simulation to decide whether a model may trade
    unattended, those numbers were describing a strategy nobody would run.

    Routing every decision through generate_trade_decision means a rule can no
    longer exist on one side only: adding one to strategy.py changes both the
    live loop and the backtest.

    Known gaps, both of which make the result optimistic:
      - Fills happen at the same bar close the decision was made on. In reality
        that close is only known once the bar has ended, so a live market order
        fills in the next bar.
      - The book is flattened on the last bar of each day rather than at
        flatten_time_et, because bar timestamps carry no timezone.
    """

    def __init__(
        self,
        model_config,
        risk_config,
        transaction_cost_bps: float,
        starting_equity: float,
        max_active_positions: int | None,
    ) -> None:
        self.model_config = model_config
        self.risk_config = risk_config
        self.cost_rate = float(transaction_cost_bps) / 10000.0
        self.starting_equity = float(starting_equity)
        self.max_active_positions = max_active_positions

        self.cash = float(starting_equity)
        self.positions: dict[str, _SimulatedPosition] = {}
        self.last_price: dict[str, float] = {}
        self.current_date = None
        self.session_start_equity = float(starting_equity)
        self.daily_trade_count = 0
        self.trade_count = 0
        self.convictions: list[int] = []
        # Per-trade detail. Aggregate metrics cannot show where an edge is
        # lost between signal and fill; only the individual fills can.
        self.open_positions: dict[str, dict] = {}
        self.closed_trades: list[dict] = []
        self.records: list[dict] = []

    def _position(self, symbol: str) -> _SimulatedPosition:
        return self.positions.setdefault(symbol, _SimulatedPosition())

    def _equity(self) -> float:
        """Cash plus holdings marked at the most recent price seen per symbol.

        A symbol can be missing from a timestamp when its features were dropped
        for that bar, so the last known price is carried forward rather than
        treating the holding as worthless.
        """
        value = self.cash
        for symbol, position in self.positions.items():
            if position.quantity:
                value += position.quantity * self.last_price.get(symbol, position.average_cost)
        return value

    def _roll_daily_state(self, trade_date) -> None:
        if self.current_date == trade_date:
            return
        self.current_date = trade_date
        self.session_start_equity = self._equity()
        self.daily_trade_count = 0

    def _daily_loss_limit_hit(self, equity: float) -> bool:
        threshold = self.session_start_equity * (1.0 - self.risk_config.max_daily_loss_pct)
        return equity <= threshold

    def _available_entry_slots(self, active_after_exits: int) -> int | None:
        slots = None
        if self.max_active_positions is not None and self.max_active_positions > 0:
            slots = max(self.max_active_positions - active_after_exits, 0)

        cap = getattr(self.risk_config, "max_daily_trade_count", None)
        if cap is not None:
            remaining = max(int(cap) - self.daily_trade_count, 0)
            slots = remaining if slots is None else min(slots, remaining)
        return slots

    def _execute(self, decision, price: float, timestamp=None) -> None:
        delta = decision.target_quantity - decision.current_quantity
        if delta == 0 or price <= 0.0:
            return

        position = self._position(decision.symbol)
        notional = abs(delta) * price

        # Buying spends cash, selling returns it; the cost is always paid out.
        self.cash -= delta * price
        self.cash -= notional * self.cost_rate

        if delta > 0:
            previous_value = position.quantity * position.average_cost
            position.quantity += delta
            position.average_cost = (previous_value + delta * price) / position.quantity
            position.bars_held = 0
            self.open_positions[decision.symbol] = {
                "symbol": decision.symbol,
                "entry_time": timestamp,
                "entry_price": price,
                "quantity": delta,
                "notional": notional,
                "conviction": decision.conviction,
                "probability_up": decision.probability_up,
            }
        else:
            opened = self.open_positions.pop(decision.symbol, None)
            position.quantity += delta
            if position.quantity <= 0:
                position.quantity = 0
                position.average_cost = 0.0
                position.bars_held = 0
            if opened is not None:
                gross = price / opened["entry_price"] - 1.0
                bars_held = None
                if timestamp is not None and opened["entry_time"] is not None:
                    bars_held = timestamp - opened["entry_time"]
                self.closed_trades.append(
                    {
                        **opened,
                        "exit_time": timestamp,
                        "exit_price": price,
                        "exit_reason": decision.reason,
                        "gross_return": gross,
                        # Cost is charged on both legs, expressed against the
                        # position so it can be compared with gross_return.
                        "net_return": gross - 2 * self.cost_rate,
                        "pnl": opened["notional"] * gross - 2 * opened["notional"] * self.cost_rate,
                        "holding": bars_held,
                    }
                )

        self.trade_count += 1
        self.daily_trade_count += 1

    def step(self, timestamp, timestamp_rows, force_flat: bool, bars_to_close=None) -> None:
        """Process every symbol quoted at one timestamp, in live-loop order."""
        for row in timestamp_rows.itertuples(index=False):
            self.last_price[row.symbol] = float(row.close)

        for position in self.positions.values():
            if position.quantity:
                position.bars_held += 1

        self._roll_daily_state(getattr(timestamp, "date", lambda: timestamp)())
        equity = self._equity()
        daily_loss_limit_hit = self._daily_loss_limit_hit(equity)

        ranked_rows = list(
            timestamp_rows.sort_values("probability_up", ascending=False).itertuples(index=False)
        )

        # First pass with entries suppressed, exactly as _run_cycle does it, so
        # the slot count reflects the positions that survive this bar's exits.
        preliminary = {}
        for row in ranked_rows:
            position = self._position(row.symbol)
            preliminary[row.symbol] = generate_trade_decision(
                symbol=row.symbol,
                probability_up=float(row.probability_up),
                last_price=float(row.close),
                current_quantity=position.quantity,
                average_cost=position.average_cost,
                equity=equity,
                model_config=self.model_config,
                risk_config=self.risk_config,
                daily_loss_limit_hit=daily_loss_limit_hit,
                allow_new_position=False,
                force_flat=force_flat,
                bars_held=self._position(row.symbol).bars_held,
                bars_to_close=bars_to_close,
            )

        active_after_exits = sum(
            1 for decision in preliminary.values() if decision.target_quantity > 0
        )
        available_slots = self._available_entry_slots(active_after_exits)

        entry_candidates = [
            row
            for row in ranked_rows
            if self._position(row.symbol).quantity == 0
            and float(row.probability_up) >= float(self.model_config.entry_probability)
        ]
        if available_slots is None:
            allowed_entries = {row.symbol for row in entry_candidates}
        else:
            allowed_entries = {row.symbol for row in entry_candidates[:available_slots]}

        for row in ranked_rows:
            position = self._position(row.symbol)
            allow_new_position = position.quantity > 0 or row.symbol in allowed_entries
            decision = generate_trade_decision(
                symbol=row.symbol,
                probability_up=float(row.probability_up),
                last_price=float(row.close),
                current_quantity=position.quantity,
                average_cost=position.average_cost,
                equity=equity,
                model_config=self.model_config,
                risk_config=self.risk_config,
                daily_loss_limit_hit=daily_loss_limit_hit,
                allow_new_position=allow_new_position,
                force_flat=force_flat,
                bars_held=position.bars_held,
                bars_to_close=bars_to_close,
            )
            if decision.action == "BUY" and decision.conviction:
                self.convictions.append(decision.conviction)
            self._execute(decision, float(row.close), timestamp)

        closing_equity = self._equity()
        invested = closing_equity - self.cash
        self.records.append(
            {
                "timestamp": timestamp,
                "equity": closing_equity,
                "active_positions": sum(
                    1 for position in self.positions.values() if position.quantity
                ),
                "exposure": invested / closing_equity if closing_equity > 0.0 else 0.0,
            }
        )


def _empty_result(entry_probability, exit_probability, max_active_positions):
    pd = _load_pandas()
    return {
        "entry_probability": float(entry_probability),
        "exit_probability": float(exit_probability),
        "max_active_positions": max_active_positions,
        "trade_count": 0,
        "mean_conviction": 0.0,
        "probability_ceiling": 1.0,
        "exposure": 0.0,
        "total_return": 0.0,
        "annualized_return": 0.0,
        "annualized_volatility": 0.0,
        "sharpe": 0.0,
        "max_drawdown": 0.0,
        "equity_curve": pd.DataFrame(columns=["timestamp", "portfolio_return", "equity_curve"]),
        "trades": pd.DataFrame(),
    }


def simulate_probability_strategy(
    prediction_rows,
    entry_probability: float,
    exit_probability: float,
    transaction_cost_bps: float = 0.0,
    max_active_positions: int | None = None,
    risk_config=None,
    starting_equity: float | None = None,
    probability_ceiling: float | None = None,
):
    """Replay predictions through the live decision rules and score the result.

    risk_config defaults to RiskConfig(), so stop loss, take profit, the daily
    loss limit and the daily trade cap are all active with the same defaults
    the paper trader ships with. Pass the risk_config actually being deployed
    to keep the two in step.
    """
    pd = _load_pandas()
    np = _load_numpy()

    required_columns = {"timestamp", "symbol", "probability_up", "close"}
    missing_columns = required_columns.difference(prediction_rows.columns)
    if missing_columns:
        raise ValueError(f"Missing columns for backtest: {sorted(missing_columns)}")

    if max_active_positions is not None and max_active_positions <= 0:
        max_active_positions = None

    if risk_config is None:
        risk_config = RiskConfig()
    if starting_equity is None:
        starting_equity = float(risk_config.starting_capital)

    if probability_ceiling is None:
        # The conviction scale has to top out inside the model's own range; see
        # strategy.conviction_score.
        probability_ceiling = float(prediction_rows["probability_up"].quantile(0.999))

    model_config = ModelConfig(
        entry_probability=float(entry_probability),
        exit_probability=float(exit_probability),
        transaction_cost_bps=float(transaction_cost_bps),
        probability_ceiling=float(probability_ceiling),
    )

    rows = prediction_rows.copy()
    rows["timestamp"] = pd.to_datetime(rows["timestamp"])
    rows = rows.sort_values(["timestamp", "symbol"]).reset_index(drop=True)
    if rows.empty:
        return _empty_result(entry_probability, exit_probability, max_active_positions)

    # The live loop flattens at flatten_time_et. Bar timestamps carry no
    # timezone, so the last bar of each day stands in for that moment here.
    last_bar_per_day = set(
        rows.groupby(rows["timestamp"].dt.date)["timestamp"].max().tolist()
    )

    simulator = _PortfolioSimulator(
        model_config=model_config,
        risk_config=risk_config,
        transaction_cost_bps=transaction_cost_bps,
        starting_equity=starting_equity,
        max_active_positions=max_active_positions,
    )
    # Bars remaining until the session's last bar, so the strategy can refuse
    # to open a position it would only have to flatten minutes later.
    unique_timestamps = sorted(rows["timestamp"].unique())
    remaining_bars = {}
    per_day: dict = {}
    for stamp in unique_timestamps:
        per_day.setdefault(pd.Timestamp(stamp).date(), []).append(stamp)
    for stamps in per_day.values():
        for offset, stamp in enumerate(stamps):
            remaining_bars[stamp] = len(stamps) - 1 - offset

    for timestamp, timestamp_rows in rows.groupby("timestamp", sort=True):
        simulator.step(
            timestamp=timestamp,
            timestamp_rows=timestamp_rows,
            force_flat=timestamp in last_bar_per_day,
            bars_to_close=remaining_bars.get(timestamp),
        )

    if not simulator.records:
        return _empty_result(entry_probability, exit_probability, max_active_positions)

    portfolio = pd.DataFrame(simulator.records)
    portfolio["equity_curve"] = portfolio["equity"] / starting_equity
    portfolio["portfolio_return"] = portfolio["equity_curve"].pct_change().fillna(
        portfolio["equity_curve"].iloc[0] - 1.0
    )

    periods_per_year = _infer_periods_per_year(portfolio["timestamp"])
    mean_return = float(portfolio["portfolio_return"].mean())
    volatility = float(portfolio["portfolio_return"].std(ddof=0))
    final_equity_ratio = float(portfolio["equity_curve"].iloc[-1])
    annualized_return = float(
        max(final_equity_ratio, 0.0) ** (periods_per_year / max(len(portfolio), 1)) - 1.0
    )
    sharpe = 0.0
    if volatility > 0.0:
        sharpe = float(mean_return / volatility * np.sqrt(periods_per_year))

    return {
        "entry_probability": float(entry_probability),
        "exit_probability": float(exit_probability),
        "probability_ceiling": float(probability_ceiling),
        "max_active_positions": max_active_positions,
        "trade_count": int(simulator.trade_count),
        "mean_conviction": float(np.mean(simulator.convictions)) if simulator.convictions else 0.0,
        "exposure": float(portfolio["exposure"].mean()),
        "total_return": final_equity_ratio - 1.0,
        "annualized_return": annualized_return,
        "annualized_volatility": float(volatility * np.sqrt(periods_per_year)),
        "sharpe": sharpe,
        "max_drawdown": _max_drawdown(portfolio["equity_curve"]),
        "equity_curve": portfolio[["timestamp", "portfolio_return", "equity_curve"]].copy(),
        "trades": pd.DataFrame(simulator.closed_trades),
    }


# A threshold that almost never fires is not a strategy, and its Sharpe is not
# evidence. Staying flat cannot lose money, so an entry bar set high enough to
# suppress trading scores well on every risk-adjusted measure while doing
# nothing - a model that learned to avoid the question rather than answer it.
# Seen live: a configuration picked at entry 0.69 traded 14 times in 54 days
# with 0.1% exposure, and its Sharpe of 0.53 went on to clear the deployment
# gate.
MINIMUM_TRADE_COUNT = 20
# Deliberately low. Its job is to catch a configuration that does not trade at
# all - the case it was written for held 0.1% exposure across 54 days - not to
# demand a heavily invested book. A selective strategy that only acts on the
# top few percent of signals, and sizes those by conviction, is legitimately
# invested only a small fraction of the time.
MINIMUM_EXPOSURE = 0.005

# Entry thresholds are searched as percentiles of the model's own probability
# distribution. An absolute grid searches a different thing for each model,
# because probability scales differ: one model's 0.60 is another's 0.72. The
# grid reaches far into the tail because that is where the edge was measured -
# the top 1% of signals realised 0.96% against 0.05% for a median qualifying
# one.
CANDIDATE_PERCENTILES = (0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99)


def _threshold_is_usable(result, min_trade_count: int, min_exposure: float) -> str | None:
    """Reason this threshold cannot be trusted, or None when it can."""
    if result["trade_count"] < min_trade_count:
        return f"only {result['trade_count']} trades"
    if result["exposure"] < min_exposure:
        return f"exposure {result['exposure']:.1%} below {min_exposure:.0%}"
    return None


def select_probability_thresholds(
    validation_rows,
    transaction_cost_bps: float,
    threshold_hysteresis: float,
    max_active_positions: int | None,
    risk_config=None,
    min_trade_count: int = MINIMUM_TRADE_COUNT,
    min_exposure: float = MINIMUM_EXPOSURE,
    candidate_percentiles=CANDIDATE_PERCENTILES,
):
    """Pick entry/exit probabilities on the validation split.

    Candidates that barely trade are rejected outright rather than ranked,
    because their scores measure inactivity rather than skill. When nothing
    qualifies, the result carries qualified=False: that a model has no
    threshold producing real activity is a finding in itself, and it should not
    be hidden behind a fallback that looks like a normal selection.
    """
    probabilities = validation_rows["probability_up"]
    # Where the conviction scale tops out, taken from the model's own range.
    probability_ceiling = float(probabilities.quantile(0.999))

    best_choice = None
    rejections: list[str] = []

    for percentile in candidate_percentiles:
        entry_probability = float(probabilities.quantile(percentile))
        exit_probability = max(entry_probability - threshold_hysteresis, 0.05)
        result = simulate_probability_strategy(
            prediction_rows=validation_rows,
            entry_probability=entry_probability,
            exit_probability=exit_probability,
            transaction_cost_bps=transaction_cost_bps,
            max_active_positions=max_active_positions,
            risk_config=risk_config,
            probability_ceiling=probability_ceiling,
        )

        rejection = _threshold_is_usable(result, min_trade_count, min_exposure)
        if rejection is not None:
            rejections.append(f"top {(1 - percentile) * 100:.0f}%: {rejection}")
            continue

        # Ranked on Sharpe then total return. The old third key nudged the
        # search towards 35% exposure, which is the opposite of what the
        # evidence supports: selectivity is where the edge lives.
        score = (result["sharpe"], result["total_return"])
        if best_choice is None or score > best_choice["score"]:
            best_choice = {
                "score": score,
                "entry_probability": entry_probability,
                "exit_probability": exit_probability,
                "entry_percentile": float(percentile),
                "probability_ceiling": probability_ceiling,
                "validation_backtest": result,
                "qualified": True,
                "selection_note": "",
            }

    if best_choice is not None:
        return best_choice

    # Nothing traded enough to be judged. Fall back to the most active
    # threshold rather than a fixed one, so the reported numbers describe the
    # closest thing to a real strategy this model can produce.
    fallback_percentile = float(candidate_percentiles[0])
    fallback_entry = float(probabilities.quantile(fallback_percentile))
    fallback_exit = max(fallback_entry - threshold_hysteresis, 0.05)
    result = simulate_probability_strategy(
        prediction_rows=validation_rows,
        entry_probability=fallback_entry,
        exit_probability=fallback_exit,
        transaction_cost_bps=transaction_cost_bps,
        max_active_positions=max_active_positions,
        risk_config=risk_config,
        probability_ceiling=probability_ceiling,
    )
    note = (
        f"no percentile reached {min_trade_count} trades and {min_exposure:.1%} exposure; "
        f"reporting the most active candidate (top {(1 - fallback_percentile) * 100:.0f}%). "
        + "; ".join(rejections[:3])
    )
    print(f"warning: {note}")
    return {
        "score": (result["sharpe"], result["total_return"]),
        "entry_probability": fallback_entry,
        "exit_probability": fallback_exit,
        "entry_percentile": fallback_percentile,
        "probability_ceiling": probability_ceiling,
        "validation_backtest": result,
        "qualified": False,
        "selection_note": note,
    }

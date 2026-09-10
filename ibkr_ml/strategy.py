from __future__ import annotations

from dataclasses import dataclass
from math import floor


# The maximum conviction score. Ten points because it reads as a
# recommendation strength rather than a probability, which it is not: the model
# is class-balanced, so its output is a ranking signal, not a calibrated chance.
MAX_CONVICTION = 10

POSITION_SIZING_MODES = ("fixed", "linear", "quadratic")


@dataclass(slots=True)
class TradeDecision:
    symbol: str
    action: str
    probability_up: float
    current_quantity: int
    target_quantity: int
    last_price: float
    reason: str
    conviction: int = 0
    position_scale: float = 1.0


def conviction_score(
    probability_up: float,
    entry_probability: float,
    probability_ceiling: float = 1.0,
) -> int:
    """Recommendation strength from 1 to 10, or 0 when the signal does not qualify.

    The old decision carried one bit of information: above the entry threshold
    or not. That discarded a real and measurable gradient - on held-out data the
    top decile of predictions realised 0.109% against 0.003% for the bottom, and
    the top 1% realised 0.96%. Buying all of them at one size charges the same
    cost against every trade, so the weak ones spend what the strong ones make.

    probability_ceiling is where the scale tops out, and it must come from the
    model's own observed range rather than from 1.0. A class-balanced gradient
    boosting model rarely prints above 0.85, so measuring headroom to 1.0 leaves
    the upper half of the scale unreachable: the strongest signals in the data
    scored 6 out of 10 and were sized as if they were average.
    """
    if probability_up < entry_probability:
        return 0

    headroom = max(probability_ceiling - entry_probability, 1e-9)
    fraction = (probability_up - entry_probability) / headroom
    return int(min(MAX_CONVICTION, 1 + int(fraction * MAX_CONVICTION)))


def position_scale(conviction: int, position_sizing: str = "fixed") -> float:
    """Fraction of the maximum position size a given conviction earns."""
    if conviction <= 0:
        return 0.0

    normalized = conviction / MAX_CONVICTION
    if position_sizing == "linear":
        return normalized
    if position_sizing == "quadratic":
        return normalized * normalized
    return 1.0


def _target_quantity(last_price: float, equity: float, risk_config, scale: float = 1.0) -> int:
    if last_price <= 0.0 or scale <= 0.0:
        return 0

    # The risk budget scales with conviction too. Sizing by conviction while
    # leaving the stop-loss budget fixed would let a low-conviction trade keep
    # the full risk allowance it no longer earns.
    risk_budget = equity * risk_config.risk_per_trade * scale
    max_notional = equity * risk_config.max_position_fraction * scale
    stop_distance = last_price * risk_config.stop_loss_pct

    if stop_distance <= 0.0:
        return 0

    quantity_from_risk = floor(risk_budget / stop_distance)
    quantity_from_notional = floor(max_notional / last_price)
    return max(min(quantity_from_risk, quantity_from_notional), 0)


def generate_trade_decision(
    symbol: str,
    probability_up: float,
    last_price: float,
    current_quantity: int,
    average_cost: float,
    equity: float,
    model_config,
    risk_config,
    daily_loss_limit_hit: bool,
    allow_new_position: bool = True,
    risk_exit_only: bool = False,
    blocked_reason: str = "risk_exit_only",
    force_flat: bool = False,
    bars_held: int = 0,
    bars_to_close: int | None = None,
) -> TradeDecision:
    """Decide what to do with one symbol for this cycle.

    When ``risk_exit_only`` is True the caller could not produce a trustworthy
    model probability for this cycle (stale bars, or the same bar as last
    cycle). Protective exits still run, because stop loss, take profit and the
    daily loss limit only depend on the position and the last price, never on
    the model. Model driven entries and exits are skipped and the decision
    falls back to HOLD with ``blocked_reason``.

    When ``force_flat`` is True the session is ending and the position has to
    go, whatever else the rules say.
    """
    if force_flat:
        # Checked ahead of every other rule, the protective stops included.
        # The model predicts three five-minute bars ahead, so a position held
        # overnight is exposed to a gap it was never trained on, for seventeen
        # hours during which no intraday stop can act on it.
        if current_quantity > 0:
            return TradeDecision(
                symbol=symbol,
                action="SELL",
                probability_up=probability_up,
                current_quantity=current_quantity,
                target_quantity=0,
                last_price=last_price,
                reason="session_close_flatten",
            )
        return TradeDecision(
            symbol=symbol,
            action="HOLD",
            probability_up=probability_up,
            current_quantity=0,
            target_quantity=0,
            last_price=last_price,
            reason="session_close_already_flat",
        )

    if current_quantity > 0 and average_cost > 0.0:
        pnl_pct = last_price / average_cost - 1.0
        if pnl_pct <= -risk_config.stop_loss_pct:
            return TradeDecision(
                symbol=symbol,
                action="SELL",
                probability_up=probability_up,
                current_quantity=current_quantity,
                target_quantity=0,
                last_price=last_price,
                reason="stop_loss",
            )
        if pnl_pct >= risk_config.take_profit_pct:
            return TradeDecision(
                symbol=symbol,
                action="SELL",
                probability_up=probability_up,
                current_quantity=current_quantity,
                target_quantity=0,
                last_price=last_price,
                reason="take_profit",
            )

    if daily_loss_limit_hit:
        if current_quantity > 0:
            return TradeDecision(
                symbol=symbol,
                action="SELL",
                probability_up=probability_up,
                current_quantity=current_quantity,
                target_quantity=0,
                last_price=last_price,
                reason="daily_loss_limit",
            )
        return TradeDecision(
            symbol=symbol,
            action="HOLD",
            probability_up=probability_up,
            current_quantity=current_quantity,
            target_quantity=current_quantity,
            last_price=last_price,
            reason="daily_loss_limit_halt",
        )

    if risk_exit_only:
        return TradeDecision(
            symbol=symbol,
            action="HOLD",
            probability_up=probability_up,
            current_quantity=current_quantity,
            target_quantity=current_quantity,
            last_price=last_price,
            reason=blocked_reason,
        )

    if current_quantity == 0 and probability_up >= model_config.entry_probability:
        close_buffer = int(getattr(risk_config, "no_entry_within_bars_of_close", 0) or 0)
        if bars_to_close is not None and bars_to_close < close_buffer:
            return TradeDecision(
                symbol=symbol,
                action="HOLD",
                probability_up=probability_up,
                current_quantity=0,
                target_quantity=0,
                last_price=last_price,
                reason="too_close_to_session_end",
            )
        if not allow_new_position:
            return TradeDecision(
                symbol=symbol,
                action="HOLD",
                probability_up=probability_up,
                current_quantity=current_quantity,
                target_quantity=current_quantity,
                last_price=last_price,
                reason="entry_filtered",
            )
        conviction = conviction_score(
            probability_up,
            model_config.entry_probability,
            getattr(model_config, "probability_ceiling", None) or 1.0,
        )
        scale = position_scale(conviction, getattr(risk_config, "position_sizing", "fixed"))
        target_quantity = _target_quantity(last_price, equity, risk_config, scale)
        action = "BUY" if target_quantity > 0 else "HOLD"
        reason = "model_entry" if target_quantity > 0 else "size_too_small"
        return TradeDecision(
            symbol=symbol,
            action=action,
            probability_up=probability_up,
            current_quantity=current_quantity,
            target_quantity=target_quantity,
            last_price=last_price,
            reason=reason,
            conviction=conviction,
            position_scale=scale,
        )

    minimum_holding = int(getattr(risk_config, "minimum_holding_bars", 0) or 0)
    if current_quantity > 0 and bars_held < minimum_holding:
        # Protective exits above already had their say; what is suppressed here
        # is only the model changing its mind before the horizon it was trained
        # on has elapsed.
        return TradeDecision(
            symbol=symbol,
            action="HOLD",
            probability_up=probability_up,
            current_quantity=current_quantity,
            target_quantity=current_quantity,
            last_price=last_price,
            reason="minimum_holding",
        )

    if current_quantity > 0 and probability_up <= model_config.exit_probability:
        return TradeDecision(
            symbol=symbol,
            action="SELL",
            probability_up=probability_up,
            current_quantity=current_quantity,
            target_quantity=0,
            last_price=last_price,
            reason="model_exit",
        )

    return TradeDecision(
        symbol=symbol,
        action="HOLD",
        probability_up=probability_up,
        current_quantity=current_quantity,
        target_quantity=current_quantity,
        last_price=last_price,
        reason="no_change",
    )

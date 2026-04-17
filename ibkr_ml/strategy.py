from __future__ import annotations

from dataclasses import dataclass
from math import floor


@dataclass(slots=True)
class TradeDecision:
    symbol: str
    action: str
    probability_up: float
    current_quantity: int
    target_quantity: int
    last_price: float
    reason: str


def _target_quantity(last_price: float, equity: float, risk_config) -> int:
    if last_price <= 0.0:
        return 0

    risk_budget = equity * risk_config.risk_per_trade
    max_notional = equity * risk_config.max_position_fraction
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
) -> TradeDecision:
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

    if current_quantity == 0 and probability_up >= model_config.entry_probability:
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
        target_quantity = _target_quantity(last_price, equity, risk_config)
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

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import connect_ib, fetch_historical_frame, load_ib_components
from .modeling import load_model_bundle, predict_probability
from .strategy import TradeDecision, generate_trade_decision


@dataclass(slots=True)
class PositionSnapshot:
    quantity: int = 0
    average_cost: float = 0.0


@dataclass(slots=True)
class SymbolSnapshot:
    symbol: str
    probability_up: float
    last_price: float
    position: PositionSnapshot
    latest_bar_timestamp: object


@dataclass(slots=True)
class CycleResult:
    decisions: list[TradeDecision]
    equity: float
    daily_loss_limit_hit: bool
    skip_reason: str | None = None


class IBKRPaperTrader:
    def __init__(
        self,
        connection_config,
        market_config,
        model_config,
        risk_config,
    ) -> None:
        self.connection_config = connection_config
        self.market_config = market_config
        self.bundle = load_model_bundle(model_config.model_path)
        trained_model_config = self.bundle.get("model_config", {})
        thresholds = self.bundle.get("thresholds", {})
        if model_config.entry_probability is None:
            model_config.entry_probability = thresholds.get("entry_probability", 0.58)
        if model_config.exit_probability is None:
            model_config.exit_probability = thresholds.get("exit_probability", 0.48)
        if getattr(risk_config, "max_active_positions", None) is None:
            risk_config.max_active_positions = trained_model_config.get("max_active_positions", 2)
        self.model_config = model_config
        self.risk_config = risk_config
        self.market_config = market_config
        self.et_zone = ZoneInfo("America/New_York")
        self.current_trade_date = None
        self.daily_trade_count = 0
        self.session_start_equity: float | None = None
        self.last_processed_bar_timestamp: dict[str, str] = {}
        self.log_dir = Path(self.risk_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_bundle()

    def _net_liquidation(self, ib) -> float:
        for item in ib.accountSummary():
            if item.tag == "NetLiquidation" and item.currency == "USD":
                return float(item.value)
        return float(self.risk_config.starting_capital)

    def _validate_model_bundle(self) -> None:
        if self.risk_config.allow_unsafe_model:
            return

        issues = []
        test_metrics = self.bundle.get("test_metrics", {})
        walk_forward = self.bundle.get("walk_forward_summary", {})

        test_auc = test_metrics.get("auc")
        if test_auc is None or test_auc < 0.60:
            issues.append(f"test_auc={test_auc}")

        fold_count = walk_forward.get("fold_count", 0)
        profitable_folds = walk_forward.get("profitable_folds", 0)
        if fold_count < 2 or profitable_folds < 2:
            issues.append(
                f"walk_forward_profitable_folds={profitable_folds}/{fold_count}"
            )

        mean_sharpe = walk_forward.get("mean_sharpe")
        if mean_sharpe is None or mean_sharpe <= 0.50:
            issues.append(f"walk_forward_mean_sharpe={mean_sharpe}")

        worst_max_drawdown = walk_forward.get("worst_max_drawdown")
        if worst_max_drawdown is None or worst_max_drawdown < -0.10:
            issues.append(f"worst_max_drawdown={worst_max_drawdown}")

        if issues:
            raise RuntimeError(
                "Model deployment gate failed: "
                + ", ".join(issues)
                + ". Use --allow-unsafe-model to override."
            )

    def _now_et(self) -> datetime:
        return datetime.now(self.et_zone)

    def _log_path(self, now_et: datetime) -> Path:
        return self.log_dir / f"paper_trade_{now_et.date().isoformat()}.jsonl"

    def _log_event(self, event_type: str, payload: dict) -> None:
        now_et = self._now_et()
        event = {
            "timestamp": now_et.isoformat(),
            "event_type": event_type,
            "payload": payload,
        }
        with self._log_path(now_et).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, default=str) + "\n")

    def _roll_daily_state(self, trade_date, equity: float) -> None:
        if self.current_trade_date == trade_date:
            return

        self.current_trade_date = trade_date
        self.session_start_equity = equity
        self.daily_trade_count = 0
        self.last_processed_bar_timestamp = {}
        self._log_event(
            "daily_reset",
            {
                "trade_date": str(trade_date),
                "session_start_equity": equity,
            },
        )

    def _within_trade_window(self, now_et: datetime) -> tuple[bool, str | None]:
        if not self.market_config.regular_session_only:
            return True, None
        if now_et.weekday() >= 5:
            return False, "weekend"

        current_time = now_et.time()
        if current_time < datetime.strptime("09:35", "%H:%M").time():
            return False, "before_open_buffer"
        if current_time > datetime.strptime("15:55", "%H:%M").time():
            return False, "after_close_buffer"
        return True, None

    def _is_stale_bar(self, latest_bar_timestamp, now_et: datetime) -> bool:
        now_naive = now_et.replace(tzinfo=None)
        latest = latest_bar_timestamp.to_pydatetime()
        age_minutes = (now_naive - latest).total_seconds() / 60.0
        return age_minutes > float(self.market_config.stale_after_minutes)

    def _blocked_decision(
        self,
        symbol: str,
        position: PositionSnapshot,
        reason: str,
        probability_up: float = 0.0,
        last_price: float = 0.0,
    ) -> TradeDecision:
        return TradeDecision(
            symbol=symbol,
            action="HOLD",
            probability_up=probability_up,
            current_quantity=position.quantity,
            target_quantity=position.quantity,
            last_price=last_price,
            reason=reason,
        )

    def _positions(self, ib):
        positions = {}
        for item in ib.positions():
            contract = item.contract
            if getattr(contract, "secType", "") != "STK":
                continue
            positions[contract.symbol] = PositionSnapshot(
                quantity=int(item.position),
                average_cost=float(item.avgCost),
            )
        return positions

    def _submit_target(self, ib, decision: TradeDecision) -> None:
        _, MarketOrder, Stock, _ = load_ib_components()

        delta = decision.target_quantity - decision.current_quantity
        if delta == 0:
            return

        contract = Stock(decision.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        action = "BUY" if delta > 0 else "SELL"
        order = MarketOrder(action, abs(delta))
        ib.placeOrder(contract, order)
        ib.sleep(1.0)
        self.daily_trade_count += 1
        self._log_event(
            "order_submitted",
            {
                "symbol": decision.symbol,
                "action": action,
                "quantity_delta": abs(delta),
                "target_quantity": decision.target_quantity,
                "current_quantity": decision.current_quantity,
                "probability_up": decision.probability_up,
                "reason": decision.reason,
            },
        )

    def _daily_loss_limit_hit(self, equity: float) -> bool:
        if self.session_start_equity is None:
            self.session_start_equity = equity
            return False

        threshold = self.session_start_equity * (1.0 - self.risk_config.max_daily_loss_pct)
        return equity <= threshold

    def _run_cycle(self, ib, dry_run: bool):
        equity = self._net_liquidation(ib)
        now_et = self._now_et()
        self._roll_daily_state(now_et.date(), equity)
        should_trade, skip_reason = self._within_trade_window(now_et)
        positions = self._positions(ib)
        daily_loss_limit_hit = self._daily_loss_limit_hit(equity)

        if not should_trade:
            result = CycleResult(
                decisions=[],
                equity=equity,
                daily_loss_limit_hit=daily_loss_limit_hit,
                skip_reason=skip_reason,
            )
            self._log_event(
                "cycle_skipped",
                {
                    "reason": skip_reason,
                    "equity": equity,
                    "daily_loss_limit_hit": daily_loss_limit_hit,
                    "daily_trade_count": self.daily_trade_count,
                },
            )
            return result

        snapshots = []
        blocked_decisions = []
        for symbol in self.market_config.symbols:
            position = positions.get(symbol, PositionSnapshot())
            try:
                frame = fetch_historical_frame(
                    ib=ib,
                    symbol=symbol,
                    duration=self.market_config.duration,
                    bar_size=self.market_config.bar_size,
                    use_rth=self.market_config.use_rth,
                    max_duration_per_request=self.market_config.max_duration_per_request,
                )
                latest_bar_timestamp = frame["timestamp"].iloc[-1]
                latest_bar_key = str(latest_bar_timestamp)
                if self._is_stale_bar(latest_bar_timestamp, now_et):
                    blocked_decisions.append(
                        self._blocked_decision(
                            symbol=symbol,
                            position=position,
                            reason="stale_data",
                            last_price=float(frame["close"].iloc[-1]),
                        )
                    )
                    continue
                if self.last_processed_bar_timestamp.get(symbol) == latest_bar_key:
                    blocked_decisions.append(
                        self._blocked_decision(
                            symbol=symbol,
                            position=position,
                            reason="same_bar",
                            last_price=float(frame["close"].iloc[-1]),
                        )
                    )
                    continue

                prediction = predict_probability(self.bundle, symbol, frame)
                snapshots.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        probability_up=prediction["probability_up"],
                        last_price=prediction["close"],
                        position=position,
                        latest_bar_timestamp=latest_bar_timestamp,
                    )
                )
            except Exception as exc:
                blocked_decisions.append(
                    self._blocked_decision(
                        symbol=symbol,
                        position=position,
                        reason=f"data_error:{exc.__class__.__name__}",
                    )
                )
                self._log_event(
                    "symbol_error",
                    {
                        "symbol": symbol,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )

        preliminary_decisions = {}
        for snapshot in snapshots:
            preliminary_decisions[snapshot.symbol] = generate_trade_decision(
                symbol=snapshot.symbol,
                probability_up=snapshot.probability_up,
                last_price=snapshot.last_price,
                current_quantity=snapshot.position.quantity,
                average_cost=snapshot.position.average_cost,
                equity=equity,
                model_config=self.model_config,
                risk_config=self.risk_config,
                daily_loss_limit_hit=daily_loss_limit_hit,
                allow_new_position=False,
            )

        active_after_exits = sum(
            1
            for decision in preliminary_decisions.values()
            if decision.target_quantity > 0
        )
        max_active_positions = getattr(self.risk_config, "max_active_positions", None)
        if max_active_positions is not None and max_active_positions <= 0:
            max_active_positions = None

        available_slots = None
        if max_active_positions is not None:
            available_slots = max(max_active_positions - active_after_exits, 0)

        remaining_trade_capacity = getattr(self.risk_config, "max_daily_trade_count", None)
        if remaining_trade_capacity is not None:
            remaining_trade_capacity = max(remaining_trade_capacity - self.daily_trade_count, 0)
            if available_slots is None:
                available_slots = remaining_trade_capacity
            else:
                available_slots = min(available_slots, remaining_trade_capacity)

        entry_candidates = [
            snapshot
            for snapshot in snapshots
            if snapshot.position.quantity == 0
            and snapshot.probability_up >= float(self.model_config.entry_probability)
        ]
        entry_candidates.sort(key=lambda item: item.probability_up, reverse=True)
        if available_slots is None:
            allowed_entry_symbols = {item.symbol for item in entry_candidates}
        else:
            allowed_entry_symbols = {item.symbol for item in entry_candidates[:available_slots]}

        decisions = []
        for snapshot in snapshots:
            allow_new_position = snapshot.position.quantity > 0 or snapshot.symbol in allowed_entry_symbols
            decision = generate_trade_decision(
                symbol=snapshot.symbol,
                probability_up=snapshot.probability_up,
                last_price=snapshot.last_price,
                current_quantity=snapshot.position.quantity,
                average_cost=snapshot.position.average_cost,
                equity=equity,
                model_config=self.model_config,
                risk_config=self.risk_config,
                daily_loss_limit_hit=daily_loss_limit_hit,
                allow_new_position=allow_new_position,
            )
            decisions.append(decision)
            self.last_processed_bar_timestamp[snapshot.symbol] = str(snapshot.latest_bar_timestamp)
            if not dry_run and decision.action in {"BUY", "SELL"}:
                self._submit_target(ib, decision)
        decisions.extend(blocked_decisions)
        decisions.sort(key=lambda item: item.symbol)
        cycle_result = CycleResult(
            decisions=decisions,
            equity=equity,
            daily_loss_limit_hit=daily_loss_limit_hit,
            skip_reason=None,
        )
        self._log_event(
            "cycle",
            {
                "equity": equity,
                "daily_loss_limit_hit": daily_loss_limit_hit,
                "daily_trade_count": self.daily_trade_count,
                "decisions": [asdict(item) for item in decisions],
                "dry_run": dry_run,
            },
        )
        return cycle_result

    def _print_cycle_result(self, result: CycleResult, dry_run: bool) -> None:
        if result.skip_reason:
            print(
                f"cycle skipped: reason={result.skip_reason} "
                f"equity={result.equity:.2f} "
                f"daily_loss_limit_hit={str(result.daily_loss_limit_hit).lower()} "
                f"daily_trade_count={self.daily_trade_count}"
            )
            return

        for decision in result.decisions:
            print(
                f"{decision.symbol}: action={decision.action} "
                f"prob_up={decision.probability_up:.3f} "
                f"qty={decision.current_quantity}->{decision.target_quantity} "
                f"price={decision.last_price:.2f} "
                f"reason={decision.reason}"
            )
        print(
            f"equity={result.equity:.2f} dry_run={str(dry_run).lower()} "
            f"daily_loss_limit_hit={str(result.daily_loss_limit_hit).lower()} "
            f"daily_trade_count={self.daily_trade_count}"
        )

    def run_once(self, dry_run: bool = False):
        ib = connect_ib(self.connection_config)
        try:
            result = self._run_cycle(ib, dry_run=dry_run)
        finally:
            ib.disconnect()

        self._print_cycle_result(result, dry_run=dry_run)
        return result.decisions

    def run_forever(self, interval_seconds: int, dry_run: bool = False):
        while True:
            try:
                self.run_once(dry_run=dry_run)
            except Exception as exc:
                self._log_event(
                    "cycle_error",
                    {
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                print(f"cycle error: {exc.__class__.__name__}: {exc}")
            finally:
                time.sleep(interval_seconds)

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from datetime import time as time_class
from pathlib import Path
from zoneinfo import ZoneInfo

from .data import (
    IB_ACKNOWLEDGED_STATES,
    IB_HELD_IN_TWS_STATES,
    connect_ib,
    fetch_historical_frame,
    load_ib_components,
    load_order_status_states,
)
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
    latest_bar_key: str


@dataclass(slots=True)
class CycleResult:
    decisions: list[TradeDecision]
    equity: float
    daily_loss_limit_hit: bool
    skip_reason: str | None = None


# US Eastern wall clock bounds of the tradable session. The start skips the
# opening auction's first minutes; the end leaves the last few minutes before
# 16:00 alone, where a market order fills at whatever the closing cross prints.
SESSION_START_ET = "09:35"
SESSION_END_ET = "15:57"


@dataclass(slots=True, frozen=True)
class SessionPhase:
    """What the loop is allowed to do at this instant of the session.

    Three phases rather than the earlier open/closed pair, because "stop
    opening positions" and "stop touching the account at all" are different
    instructions and the window between them is what closes the book.
    """

    name: str
    skip_reason: str | None = None

    @property
    def is_closed(self) -> bool:
        return self.name == "closed"

    @property
    def force_flat(self) -> bool:
        return self.name == "flatten"


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
        # Bars come back from TWS/IB Gateway in that machine's local timezone,
        # which is US Eastern only by default; see MarketDataConfig.bar_timezone.
        bar_timezone = getattr(market_config, "bar_timezone", None)
        self.bar_zone = ZoneInfo(bar_timezone) if bar_timezone else self.et_zone
        self.current_trade_date = None
        self.daily_trade_count = 0
        self.session_start_equity: float | None = None
        self.last_processed_bar_timestamp: dict[str, str] = {}
        # Reference series this model was trained against, e.g. {"mkt": "SPY"}.
        # Empty for a model trained without cross-asset features.
        self.reference_symbols = dict(self.bundle.get("reference_symbols") or {})
        self._order_states_cache: frozenset[str] | None = None
        # How long to wait for IBKR to acknowledge an order before treating
        # it as held inside TWS.
        self.order_acknowledgement_timeout = 5.0
        self.log_dir = Path(self.risk_config.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._validate_model_bundle()
        self._warn_on_timezone_mismatch()

    def _warn_on_timezone_mismatch(self) -> None:
        """Report when the model was trained on bars read in another timezone.

        Features are converted to US Eastern on both sides, so a declared
        mismatch is harmless. An undeclared one is not: a bundle trained
        without --bar-timezone on a UTC host carries time-of-day features
        shifted by the UTC offset, and scoring it correctly here would feed the
        model numbers it never saw in training.
        """
        if "bar_timezone" not in self.bundle:
            print(
                "note: this model bundle predates timezone-aware features. "
                "Retrain it if TWS does not run on US Eastern time."
            )
            return

        trained_zone = self.bundle.get("bar_timezone")
        live_zone = getattr(self.market_config, "bar_timezone", None)
        if trained_zone != live_zone:
            print(
                f"note: model trained on bars read as {trained_zone or 'US/Eastern'}, "
                f"scoring bars read as {live_zone or 'US/Eastern'}. Both are "
                "converted to US Eastern, so this is only a problem if either "
                "declaration is wrong."
            )

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

    def _flatten_time(self) -> time_class:
        """Parse the configured flatten time, falling back on a bad value.

        A typo here would otherwise either crash the loop mid-session or, worse,
        silently disable the flatten and carry the book overnight.
        """
        raw = str(getattr(self.risk_config, "flatten_time_et", "15:45"))
        try:
            return datetime.strptime(raw, "%H:%M").time()
        except ValueError:
            print(
                f"warning: flatten_time_et={raw!r} is not HH:MM, "
                "falling back to 15:45"
            )
            return datetime.strptime("15:45", "%H:%M").time()

    def _session_phase(self, now_et: datetime) -> SessionPhase:
        """Which phase of the session now_et falls in.

        trading -> entries and exits both allowed
        flatten -> exits only, every open position is closed
        closed  -> the loop does nothing
        """
        if not self.market_config.regular_session_only:
            return SessionPhase("trading")
        if now_et.weekday() >= 5:
            return SessionPhase("closed", "weekend")

        current_time = now_et.time()
        if current_time < datetime.strptime(SESSION_START_ET, "%H:%M").time():
            return SessionPhase("closed", "before_open_buffer")
        if current_time > datetime.strptime(SESSION_END_ET, "%H:%M").time():
            return SessionPhase("closed", "after_close_buffer")

        if getattr(self.risk_config, "flatten_before_close", True):
            if current_time >= self._flatten_time():
                return SessionPhase("flatten")
        return SessionPhase("trading")

    def _normalize_bar_timestamp(self, latest_bar_timestamp) -> datetime:
        """Return the bar timestamp as an aware datetime in US Eastern.

        A naive timestamp is stamped with self.bar_zone, the timezone of the
        machine running TWS/IB Gateway. Stamping it with the wrong zone shifts
        every age computation by the offset between the two zones.
        """
        if hasattr(latest_bar_timestamp, "to_pydatetime"):
            latest = latest_bar_timestamp.to_pydatetime()
        else:
            latest = latest_bar_timestamp

        if latest.tzinfo is None:
            return latest.replace(tzinfo=self.bar_zone).astimezone(self.et_zone)
        return latest.astimezone(self.et_zone)

    def _bar_age_minutes(self, latest_bar_et: datetime, now_et: datetime) -> float:
        return (now_et - latest_bar_et).total_seconds() / 60.0

    def _staleness_reason(self, age_minutes: float) -> str | None:
        """Reason to distrust this bar, or None when it is usable.

        The negative side matters as much as the positive one: a timestamp in
        the future means the timezone was misread, and with only an upper bound
        such an age never exceeds the threshold, which silently disables the
        staleness guard instead of reporting a fault.
        """
        tolerance = float(getattr(self.market_config, "future_bar_tolerance_minutes", 1.0))
        if age_minutes < -tolerance:
            return "future_bar_timestamp"
        if age_minutes > float(self.market_config.stale_after_minutes):
            return "stale_data"
        return None

    def _fetch_reference_frames(self, ib, now_et: datetime):
        """Fetch the reference series this model needs, or report why it cannot.

        Returns (frames, blocked_reason). A model trained with cross-asset
        features cannot be scored without them: reindex would fill the missing
        columns with zeroes, and zero is a perfectly meaningful value for a
        return - the model would read "the market did not move" rather than
        "unknown". So a missing or stale reference blocks model scoring for
        every symbol this cycle. Protective exits still run, because they never
        touch the model.
        """
        if not self.reference_symbols:
            return {}, None

        frames = {}
        for key, symbol in self.reference_symbols.items():
            try:
                frame = fetch_historical_frame(
                    ib=ib,
                    symbol=symbol,
                    duration=self.market_config.duration,
                    bar_size=self.market_config.bar_size,
                    use_rth=self.market_config.use_rth,
                    max_duration_per_request=self.market_config.max_duration_per_request,
                )
            except Exception as exc:
                reason = f"reference_{key}_{self._data_error_reason(exc)}"
                self._log_event(
                    "reference_error",
                    {
                        "kind": key,
                        "symbol": symbol,
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                return None, reason

            latest_bar_et = self._normalize_bar_timestamp(frame["timestamp"].iloc[-1])
            age_minutes = self._bar_age_minutes(latest_bar_et, now_et)
            staleness = self._staleness_reason(age_minutes)
            if staleness is not None:
                self._log_event(
                    "reference_blocked",
                    {
                        "kind": key,
                        "symbol": symbol,
                        "reason": staleness,
                        "latest_bar_et": latest_bar_et.isoformat(),
                        "now_et": now_et.isoformat(),
                        "age_minutes": round(age_minutes, 3),
                    },
                )
                return None, f"reference_{key}_{staleness}"

            frames[key] = frame
        return frames, None

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

    def _data_error_reason(self, exc: Exception) -> str:
        """Build a stable reason string from IBKR's structured error codes.

        Reads the codes the data layer attached to IBDataError instead of
        pattern matching a human readable message, which silently broke on any
        format change and matched "code 1620" as "code 162". original_type
        keeps the reason stable across the exception wrapping in data.py.
        """
        if getattr(exc, "is_session_conflict", False):
            return "data_error:IBKR_162_different_ip"

        codes = getattr(exc, "codes", None)
        if codes:
            return f"data_error:IBKR_{codes[0]}"

        original_type = getattr(exc, "original_type", None) or exc.__class__.__name__
        return f"data_error:{original_type}"

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

    @property
    def active_order_states(self) -> frozenset[str]:
        """Order statuses that mean an order is still working, loaded lazily.

        Reading this at construction time would make ib_insync a hard import
        for anything that only builds a trader to inspect its decisions.
        """
        if self._order_states_cache is None:
            self._order_states_cache = load_order_status_states()
        return self._order_states_cache

    def _active_trades_for_symbol(self, ib, symbol: str) -> list:
        """Orders still working at IBKR for one symbol.

        openTrades() can also return trades that reached a terminal state
        during this session, so the status is checked rather than assumed.
        """
        trades = []
        for trade in ib.openTrades():
            contract = getattr(trade, "contract", None)
            if getattr(contract, "symbol", None) != symbol:
                continue
            status = str(getattr(getattr(trade, "orderStatus", None), "status", ""))
            if status in self.active_order_states:
                trades.append(trade)
        return trades

    def _has_working_entry_order(self, ib, symbol: str) -> bool:
        """True when an unfilled entry order for this symbol is still in flight.

        Without this the loop reads a position of zero from IBKR while its own
        market order is still working, decides to enter again, and ends up with
        twice the intended size. Only a market parent counts: the stop and
        limit children of a bracket are meant to sit there while the position
        is open, and treating them as in-flight entries would freeze the symbol.
        """
        for trade in self._active_trades_for_symbol(ib, symbol):
            order_type = str(getattr(getattr(trade, "order", None), "orderType", ""))
            if order_type == "MKT":
                return True
        return False

    def _cancel_open_orders(self, ib, symbol: str) -> list[int]:
        """Cancel every working order for symbol, returning the ids cancelled.

        A protective bracket child sits at IBKR waiting to sell the position.
        Sending a separate closing order without cancelling it first fills
        twice - once from the closing order, once from the stop or the take
        profit - which leaves a short position the strategy never asked for.
        """
        cancelled_ids: list[int] = []
        for trade in self._active_trades_for_symbol(ib, symbol):
            order = getattr(trade, "order", None)
            if order is None:
                continue
            try:
                ib.cancelOrder(order)
            except Exception as exc:
                self._log_event(
                    "order_cancel_failed",
                    {
                        "symbol": symbol,
                        "order_id": getattr(order, "orderId", None),
                        "order_type": getattr(order, "orderType", None),
                        "error_type": exc.__class__.__name__,
                        "error": str(exc),
                    },
                )
                continue
            cancelled_ids.append(int(getattr(order, "orderId", 0)))

        if cancelled_ids:
            # Let IBKR acknowledge the cancellations before the closing order
            # goes out, so the two cannot both be live at the same instant.
            ib.sleep(1.0)
        return cancelled_ids

    def bracket_protection_prices(self, reference_price: float) -> tuple[float, float]:
        """Stop and take-profit prices for a bracket around reference_price.

        A market entry has no known fill price while the order is being built,
        so the bracket is anchored on the last bar close. The realized stop
        distance therefore differs from stop_loss_pct by the slippage between
        that close and the actual fill, which is why the reference price is
        logged next to both levels.
        """
        stop_price = reference_price * (1.0 - self.risk_config.stop_loss_pct)
        take_profit_price = reference_price * (1.0 + self.risk_config.take_profit_pct)
        # US equities above one dollar trade in one cent increments; an
        # unrounded price is rejected by IBKR as a bad tick size.
        return round(stop_price, 2), round(take_profit_price, 2)

    def _submit_bracket_entry(self, ib, contract, decision: TradeDecision, quantity: int) -> dict:
        """Place a market entry with server-side stop-loss and take-profit children.

        Only the last order carries transmit=True: IBKR holds the whole group
        until it arrives, then activates the children as soon as the parent
        fills. The children are GTC so they survive to the next session if a
        position is ever held overnight; every close path cancels them first.
        """
        components = load_ib_components()
        stop_price, take_profit_price = self.bracket_protection_prices(decision.last_price)

        parent = components.MarketOrder(
            "BUY",
            quantity,
            orderId=ib.client.getReqId(),
            transmit=False,
        )
        take_profit = components.LimitOrder(
            "SELL",
            quantity,
            take_profit_price,
            orderId=ib.client.getReqId(),
            transmit=False,
            parentId=parent.orderId,
            tif="GTC",
        )
        stop_loss = components.StopOrder(
            "SELL",
            quantity,
            stop_price,
            orderId=ib.client.getReqId(),
            transmit=True,
            parentId=parent.orderId,
            tif="GTC",
        )

        placed = [ib.placeOrder(contract, order) for order in (parent, take_profit, stop_loss)]

        return {
            "trade": placed[0],
            "order_style": "bracket",
            "reference_price": decision.last_price,
            "stop_price": stop_price,
            "take_profit_price": take_profit_price,
            "parent_order_id": parent.orderId,
            "take_profit_order_id": take_profit.orderId,
            "stop_loss_order_id": stop_loss.orderId,
        }

    def _await_order_acknowledgement(self, ib, trade, timeout_seconds: float | None = None) -> str:
        """Wait until IBKR acknowledges the order, and return the status reached.

        placeOrder returns straight away; the order has only really left when
        TWS has passed it on. One that stays in PendingSubmit was held back by
        TWS itself - most often by the order precautions dialog waiting for a
        human click - and vanishes when the client disconnects. Counting that as
        a submitted order is how a log ends up claiming trades that never
        happened.

        A trade object is not always available (a stubbed broker, an older
        ib_insync), in which case there is nothing to check and the previous
        fixed sleep is the best that can be done.
        """
        if timeout_seconds is None:
            timeout_seconds = self.order_acknowledgement_timeout
        if getattr(trade, "orderStatus", None) is None:
            ib.sleep(1.0)
            return "unknown"

        deadline = time.monotonic() + timeout_seconds
        previous_status = None
        while True:
            status = str(getattr(trade.orderStatus, "status", "") or "")
            if status in IB_ACKNOWLEDGED_STATES:
                return status
            if status and status not in IB_HELD_IN_TWS_STATES and status == previous_status:
                # A terminal state that survived a second read. One read is not
                # enough: a TWS order preset can bounce an order through
                # Cancelled and back to Submitted inside the same second - seen
                # live as "Error 10349: Order TIF was set to DAY based on order
                # preset" followed by a fill - and reporting that first read
                # would call a filled order rejected.
                return status
            if time.monotonic() >= deadline:
                return status or "unknown"
            previous_status = status
            ib.sleep(0.5)

    def _submit_target(self, ib, decision: TradeDecision) -> None:
        components = load_ib_components()

        delta = decision.target_quantity - decision.current_quantity
        if delta == 0:
            return

        if self._has_working_entry_order(ib, decision.symbol):
            self._log_event(
                "order_skipped",
                {
                    "symbol": decision.symbol,
                    "reason": "entry_order_already_working",
                    "quantity_delta": abs(delta),
                    "decision_reason": decision.reason,
                },
            )
            return

        contract = components.Stock(decision.symbol, "SMART", "USD")
        ib.qualifyContracts(contract)
        action = "BUY" if delta > 0 else "SELL"

        cancelled_order_ids: list[int] = []
        if delta < 0:
            cancelled_order_ids = self._cancel_open_orders(ib, decision.symbol)

        opens_new_position = delta > 0 and decision.current_quantity == 0
        if opens_new_position and self.risk_config.use_bracket_orders:
            order_detail = self._submit_bracket_entry(ib, contract, decision, abs(delta))
        else:
            trade = ib.placeOrder(contract, components.MarketOrder(action, abs(delta)))
            order_detail = {"order_style": "market", "trade": trade}

        trade = order_detail.pop("trade", None)
        status = self._await_order_acknowledgement(ib, trade)

        payload = {
            "symbol": decision.symbol,
            "action": action,
            "quantity_delta": abs(delta),
            "target_quantity": decision.target_quantity,
            "current_quantity": decision.current_quantity,
            "probability_up": decision.probability_up,
            "reason": decision.reason,
            "cancelled_order_ids": cancelled_order_ids,
            "order_status": status,
            **order_detail,
        }

        if status in IB_HELD_IN_TWS_STATES:
            # Not counted against the daily trade cap: nothing was traded, and
            # spending the budget on an order TWS never sent would silently
            # shrink the day's remaining capacity.
            self._log_event("order_not_transmitted", payload)
            print(
                f"warning: {decision.symbol} {action} order is stuck in "
                f"{status}; TWS never passed it to IBKR. Check Global "
                "Configuration - API - Precautions - 'Bypass Order Precautions "
                "for API Orders', and look for a dialog waiting in TWS."
            )
            return

        self.daily_trade_count += 1
        self._log_event("order_submitted", payload)

    def _daily_loss_limit_hit(self, equity: float) -> bool:
        if self.session_start_equity is None:
            self.session_start_equity = equity
            return False

        threshold = self.session_start_equity * (1.0 - self.risk_config.max_daily_loss_pct)
        return equity <= threshold

    def _flatten_cycle(self, ib, positions, equity: float, daily_loss_limit_hit: bool, dry_run: bool):
        """Close every position this strategy holds, without asking the model.

        Deliberately skips the historical data request. Closing the book is not
        negotiable, so a stale bar or a failed download must not be able to
        leave a position open overnight - which is exactly what happened when
        the end of the session was a plain skip. last_price is therefore
        unknown here, which costs nothing: a flatten targets zero shares and
        never needs a size.

        Only symbols this strategy trades are touched. Anything else in the
        account was put there by someone else and is none of its business.
        """
        decisions = []
        for symbol in self.market_config.symbols:
            position = positions.get(symbol, PositionSnapshot())
            decision = generate_trade_decision(
                symbol=symbol,
                probability_up=0.0,
                last_price=0.0,
                current_quantity=position.quantity,
                average_cost=position.average_cost,
                equity=equity,
                model_config=self.model_config,
                risk_config=self.risk_config,
                daily_loss_limit_hit=daily_loss_limit_hit,
                force_flat=True,
            )
            decisions.append(decision)
            if not dry_run and decision.action == "SELL":
                self._submit_target(ib, decision)

        decisions.sort(key=lambda item: item.symbol)
        self._log_event(
            "cycle",
            {
                "phase": "flatten",
                "equity": equity,
                "daily_loss_limit_hit": daily_loss_limit_hit,
                "daily_trade_count": self.daily_trade_count,
                "decisions": [asdict(item) for item in decisions],
                "dry_run": dry_run,
            },
        )
        return CycleResult(
            decisions=decisions,
            equity=equity,
            daily_loss_limit_hit=daily_loss_limit_hit,
            skip_reason=None,
        )

    def _run_cycle(self, ib, dry_run: bool):
        equity = self._net_liquidation(ib)
        now_et = self._now_et()
        self._roll_daily_state(now_et.date(), equity)
        phase = self._session_phase(now_et)
        positions = self._positions(ib)
        daily_loss_limit_hit = self._daily_loss_limit_hit(equity)

        if phase.is_closed:
            result = CycleResult(
                decisions=[],
                equity=equity,
                daily_loss_limit_hit=daily_loss_limit_hit,
                skip_reason=phase.skip_reason,
            )
            self._log_event(
                "cycle_skipped",
                {
                    "reason": phase.skip_reason,
                    "equity": equity,
                    "daily_loss_limit_hit": daily_loss_limit_hit,
                    "daily_trade_count": self.daily_trade_count,
                },
            )
            return result

        if phase.force_flat:
            return self._flatten_cycle(
                ib=ib,
                positions=positions,
                equity=equity,
                daily_loss_limit_hit=daily_loss_limit_hit,
                dry_run=dry_run,
            )

        reference_frames, reference_blocked_reason = self._fetch_reference_frames(ib, now_et)

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
                latest_bar_et = self._normalize_bar_timestamp(latest_bar_timestamp)
                latest_bar_key = latest_bar_et.isoformat()
                last_price = float(frame["close"].iloc[-1])
                age_minutes = self._bar_age_minutes(latest_bar_et, now_et)

                blocked_reason = self._staleness_reason(age_minutes)
                if blocked_reason is None and reference_blocked_reason is not None:
                    blocked_reason = reference_blocked_reason
                if (
                    blocked_reason is None
                    and self.last_processed_bar_timestamp.get(symbol) == latest_bar_key
                ):
                    blocked_reason = "same_bar"

                if blocked_reason is not None:
                    # Record what the staleness check actually saw. Without
                    # these fields a run full of stale_data cannot be told
                    # apart from a delayed feed, a misread timezone, or a
                    # stale_after_minutes set too low.
                    self._log_event(
                        "bar_blocked",
                        {
                            "symbol": symbol,
                            "reason": blocked_reason,
                            "latest_bar_raw": str(latest_bar_timestamp),
                            "latest_bar_et": latest_bar_key,
                            "now_et": now_et.isoformat(),
                            "age_minutes": round(age_minutes, 3),
                            "stale_after_minutes": self.market_config.stale_after_minutes,
                            "bar_timezone": str(self.bar_zone),
                            "last_price": last_price,
                            "price_is_stale": blocked_reason != "same_bar",
                        },
                    )
                    # Protective exits still run. Stop loss, take profit and the
                    # daily loss limit depend only on the position and the last
                    # price, never on the model, so skipping the whole symbol
                    # here used to mean a held position was never checked
                    # against its stop.
                    blocked_decisions.append(
                        generate_trade_decision(
                            symbol=symbol,
                            probability_up=0.0,
                            last_price=last_price,
                            current_quantity=position.quantity,
                            average_cost=position.average_cost,
                            equity=equity,
                            model_config=self.model_config,
                            risk_config=self.risk_config,
                            daily_loss_limit_hit=daily_loss_limit_hit,
                            allow_new_position=False,
                            risk_exit_only=True,
                            blocked_reason=blocked_reason,
                        )
                    )
                    continue

                prediction = predict_probability(
                    self.bundle,
                    symbol,
                    frame,
                    bar_timezone=getattr(self.market_config, "bar_timezone", None),
                    reference_frames=reference_frames or None,
                )
                snapshots.append(
                    SymbolSnapshot(
                        symbol=symbol,
                        probability_up=prediction["probability_up"],
                        last_price=prediction["close"],
                        position=position,
                        latest_bar_key=latest_bar_key,
                    )
                )
            except Exception as exc:
                blocked_decisions.append(
                    self._blocked_decision(
                        symbol=symbol,
                        position=position,
                        reason=self._data_error_reason(exc),
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
            self.last_processed_bar_timestamp[snapshot.symbol] = snapshot.latest_bar_key
            if not dry_run and decision.action in {"BUY", "SELL"}:
                self._submit_target(ib, decision)
        decisions.extend(blocked_decisions)
        if not dry_run:
            # Blocked symbols can still produce a protective SELL, so these
            # decisions must reach the broker like the normal ones.
            for decision in blocked_decisions:
                if decision.action in {"BUY", "SELL"}:
                    self._submit_target(ib, decision)
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
                "phase": "trading",
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
        try:
            while True:
                cycle_started = time.monotonic()
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

                # Sleep outside finally, so Ctrl-C stops the loop immediately
                # instead of waiting out a whole interval, and subtract the
                # cycle's own runtime so the schedule does not drift away from
                # the bar boundaries.
                elapsed_seconds = time.monotonic() - cycle_started
                time.sleep(max(float(interval_seconds) - elapsed_seconds, 0.0))
        except KeyboardInterrupt:
            print("stopped by user")

"""Execution: session phases, order placement, and the freshness guard.

The order tests assert on what reached the fake broker, because "the decision
said SELL" and "a sell order was actually sent, and the protective orders that
would have sold a second time were cancelled first" are different claims.
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from conftest import FakeContract, FakeIB, FakeOrder, FakeOrderStatus, FakePosition, FakeTrade
from ibkr_ml.config import MarketDataConfig, ModelConfig, RiskConfig
from ibkr_ml.strategy import TradeDecision

ET = ZoneInfo("America/New_York")


def et(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=ET)


class TestSessionPhase:
    def test_midday_is_a_trading_phase(self, make_trader):
        trader = make_trader()
        phase = trader._session_phase(et("2026-01-05 11:00"))
        assert phase.name == "trading"
        assert not phase.force_flat
        assert not phase.is_closed

    def test_before_the_open_buffer_is_closed(self, make_trader):
        trader = make_trader()
        phase = trader._session_phase(et("2026-01-05 09:20"))
        assert phase.is_closed
        assert phase.skip_reason == "before_open_buffer"

    def test_after_the_close_buffer_is_closed(self, make_trader):
        trader = make_trader()
        phase = trader._session_phase(et("2026-01-05 16:30"))
        assert phase.is_closed
        assert phase.skip_reason == "after_close_buffer"

    def test_weekend_is_closed(self, make_trader):
        trader = make_trader()
        phase = trader._session_phase(et("2026-01-10 11:00"))  # a Saturday
        assert phase.is_closed
        assert phase.skip_reason == "weekend"

    def test_the_last_minutes_are_a_flatten_phase(self, make_trader):
        trader = make_trader()
        phase = trader._session_phase(et("2026-01-05 15:50"))
        assert phase.force_flat
        assert not phase.is_closed

    def test_flatten_starts_exactly_at_the_configured_time(self, make_trader):
        trader = make_trader(risk_config=RiskConfig(flatten_time_et="15:30"))
        assert trader._session_phase(et("2026-01-05 15:29")).name == "trading"
        assert trader._session_phase(et("2026-01-05 15:30")).force_flat

    def test_flatten_can_be_switched_off(self, make_trader):
        trader = make_trader(risk_config=RiskConfig(flatten_before_close=False))
        assert trader._session_phase(et("2026-01-05 15:50")).name == "trading"

    def test_a_malformed_flatten_time_falls_back_instead_of_crashing(self, make_trader, capsys):
        trader = make_trader(risk_config=RiskConfig(flatten_time_et="not a time"))
        phase = trader._session_phase(et("2026-01-05 15:50"))
        assert phase.force_flat
        assert "falling back to 15:45" in capsys.readouterr().out


class TestBracketOrders:
    def test_a_new_position_is_sent_as_a_bracket(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        decision = TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry")

        trader._submit_target(ib, decision)

        assert len(ib.placed) == 3
        order_types = [order.orderType for _, order in ib.placed]
        assert order_types == ["MKT", "LMT", "STP"]

    def test_the_children_hang_off_the_parent_and_only_the_last_transmits(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        trader._submit_target(ib, TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))

        parent, take_profit, stop_loss = (order for _, order in ib.placed)
        assert take_profit.parentId == parent.orderId
        assert stop_loss.parentId == parent.orderId
        assert (parent.transmit, take_profit.transmit, stop_loss.transmit) == (False, False, True)

    def test_protection_prices_bracket_the_reference_price(self, make_trader):
        trader = make_trader(risk_config=RiskConfig(stop_loss_pct=0.01, take_profit_pct=0.02))
        stop_price, take_profit_price = trader.bracket_protection_prices(200.0)
        assert stop_price == 198.0
        assert take_profit_price == 204.0

    def test_protection_prices_are_rounded_to_the_cent(self, make_trader):
        trader = make_trader()
        stop_price, take_profit_price = trader.bracket_protection_prices(333.33)
        assert stop_price == round(stop_price, 2)
        assert take_profit_price == round(take_profit_price, 2)

    def test_the_bracket_is_logged_with_both_levels(self, make_trader, read_log_events):
        trader = make_trader()
        trader._submit_target(FakeIB(), TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))

        submitted = [event for event in read_log_events() if event["event_type"] == "order_submitted"]
        assert len(submitted) == 1
        payload = submitted[0]["payload"]
        assert payload["order_style"] == "bracket"
        assert payload["stop_price"] < payload["reference_price"] < payload["take_profit_price"]

    def test_brackets_can_be_switched_off(self, make_trader):
        trader = make_trader(risk_config=RiskConfig(use_bracket_orders=False))
        ib = FakeIB()
        trader._submit_target(ib, TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))
        assert len(ib.placed) == 1
        assert ib.placed[0][1].orderType == "MKT"

    def test_closing_a_position_sends_one_plain_market_order(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))
        assert len(ib.placed) == 1
        _, order = ib.placed[0]
        assert (order.orderType, order.action, order.totalQuantity) == ("MKT", "SELL", 100)


class TestProtectiveOrderCancellation:
    def make_ib_holding_a_bracket(self):
        contract = FakeContract("AAA")
        return FakeIB(
            open_trades=[
                FakeTrade(contract, FakeOrder(11, "STP"), FakeOrderStatus("Submitted")),
                FakeTrade(contract, FakeOrder(12, "LMT"), FakeOrderStatus("Submitted")),
            ]
        )

    def test_closing_cancels_the_protective_children_first(self, make_trader):
        trader = make_trader()
        ib = self.make_ib_holding_a_bracket()

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        assert sorted(order.orderId for order in ib.cancelled) == [11, 12]
        assert len(ib.placed) == 1

    def test_another_symbols_orders_are_left_alone(self, make_trader):
        trader = make_trader()
        ib = FakeIB(
            open_trades=[
                FakeTrade(FakeContract("BBB"), FakeOrder(21, "STP"), FakeOrderStatus("Submitted")),
            ]
        )
        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))
        assert ib.cancelled == []

    def test_terminated_orders_are_not_cancelled_again(self, make_trader):
        trader = make_trader()
        ib = FakeIB(
            open_trades=[
                FakeTrade(FakeContract("AAA"), FakeOrder(31, "STP"), FakeOrderStatus("Filled")),
            ]
        )
        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))
        assert ib.cancelled == []

    def test_a_failed_cancellation_is_logged_and_the_close_still_goes_out(
        self, make_trader, read_log_events
    ):
        trader = make_trader()
        ib = self.make_ib_holding_a_bracket()
        ib.cancel_should_raise = True

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        failures = [e for e in read_log_events() if e["event_type"] == "order_cancel_failed"]
        assert len(failures) == 2
        assert len(ib.placed) == 1

    def test_opening_a_position_does_not_cancel_anything(self, make_trader):
        trader = make_trader()
        ib = self.make_ib_holding_a_bracket()
        trader._submit_target(ib, TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))
        assert ib.cancelled == []


class TestDuplicateOrderGuard:
    def test_an_unfilled_market_order_blocks_a_second_one(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(
            open_trades=[
                FakeTrade(FakeContract("AAA"), FakeOrder(41, "MKT"), FakeOrderStatus("PreSubmitted")),
            ]
        )

        trader._submit_target(ib, TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))

        assert ib.placed == []
        skipped = [e for e in read_log_events() if e["event_type"] == "order_skipped"]
        assert skipped[0]["payload"]["reason"] == "entry_order_already_working"

    def test_resting_protective_orders_do_not_block_a_new_order(self, make_trader):
        trader = make_trader()
        ib = FakeIB(
            open_trades=[
                FakeTrade(FakeContract("AAA"), FakeOrder(51, "STP"), FakeOrderStatus("Submitted")),
            ]
        )
        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))
        assert len(ib.placed) == 1

    def test_a_no_op_decision_places_nothing(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        trader._submit_target(ib, TradeDecision("AAA", "HOLD", 0.5, 100, 100, 50.0, "no_change"))
        assert ib.placed == []


class TestFlattenCycle:
    def test_every_held_symbol_is_closed(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB()
        positions = {
            "AAA": type("P", (), {"quantity": 100, "average_cost": 50.0})(),
            "BBB": type("P", (), {"quantity": 40, "average_cost": 20.0})(),
        }

        result = trader._flatten_cycle(ib, positions, 100000.0, False, dry_run=False)

        assert {d.symbol for d in result.decisions if d.action == "SELL"} == {"AAA", "BBB"}
        assert len(ib.placed) == 2
        assert all(order.action == "SELL" for _, order in ib.placed)

    def test_a_flat_book_places_no_orders(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        result = trader._flatten_cycle(ib, {}, 100000.0, False, dry_run=False)
        assert ib.placed == []
        assert all(d.reason == "session_close_already_flat" for d in result.decisions)

    def test_dry_run_decides_but_sends_nothing(self, make_trader):
        trader = make_trader()
        ib = FakeIB()
        positions = {"AAA": type("P", (), {"quantity": 100, "average_cost": 50.0})()}
        result = trader._flatten_cycle(ib, positions, 100000.0, False, dry_run=True)
        assert any(d.action == "SELL" for d in result.decisions)
        assert ib.placed == []

    def test_symbols_outside_the_strategy_are_untouched(self, make_trader):
        trader = make_trader(market_config=MarketDataConfig(symbols=("AAA",)))
        ib = FakeIB()
        positions = {
            "AAA": type("P", (), {"quantity": 100, "average_cost": 50.0})(),
            "ZZZ": type("P", (), {"quantity": 999, "average_cost": 10.0})(),
        }
        result = trader._flatten_cycle(ib, positions, 100000.0, False, dry_run=False)
        assert {d.symbol for d in result.decisions} == {"AAA"}
        assert len(ib.placed) == 1


class TestFreshnessGuard:
    def test_a_fresh_bar_passes(self, make_trader):
        trader = make_trader()
        assert trader._staleness_reason(3.0) is None

    def test_an_old_bar_is_stale(self, make_trader):
        trader = make_trader()
        assert trader._staleness_reason(20.0) == "stale_data"

    def test_a_future_bar_is_reported_rather_than_accepted(self, make_trader):
        trader = make_trader()
        assert trader._staleness_reason(-30.0) == "future_bar_timestamp"

    def test_a_slightly_future_bar_is_tolerated(self, make_trader):
        trader = make_trader()
        assert trader._staleness_reason(-0.5) is None

    def test_naive_bars_are_read_in_the_configured_zone(self, make_trader):
        trader = make_trader(
            market_config=MarketDataConfig(symbols=("AAA",), bar_timezone="UTC")
        )
        normalized = trader._normalize_bar_timestamp(datetime(2026, 1, 5, 20, 0))
        # 20:00 UTC in January is 15:00 US Eastern.
        assert normalized.hour == 15
        assert normalized.tzinfo is not None

    def test_naive_bars_default_to_us_eastern(self, make_trader):
        trader = make_trader()
        normalized = trader._normalize_bar_timestamp(datetime(2026, 1, 5, 15, 0))
        assert normalized.hour == 15


class TestDeploymentGate:
    def test_a_weak_model_is_refused(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["test_metrics"]["auc"] = 0.51
        with pytest.raises(RuntimeError, match="Model deployment gate failed"):
            make_trader(bundle=bundle)

    def test_unprofitable_walk_forward_folds_are_refused(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["walk_forward_summary"]["profitable_folds"] = 1
        with pytest.raises(RuntimeError, match="walk_forward_profitable_folds"):
            make_trader(bundle=bundle)

    def test_the_override_flag_lets_a_weak_model_through(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["test_metrics"]["auc"] = 0.10
        trader = make_trader(bundle=bundle, risk_config=RiskConfig(allow_unsafe_model=True))
        assert trader is not None


class TestDataErrorReason:
    def test_a_session_conflict_is_named(self, make_trader):
        from ibkr_ml.data import IBDataError

        trader = make_trader()
        error = IBDataError(
            "boom",
            ib_errors=[{"code": 162, "message": "already connected from a different IP address"}],
        )
        assert trader._data_error_reason(error) == "data_error:IBKR_162_different_ip"

    def test_a_plain_ibkr_code_is_carried_through(self, make_trader):
        from ibkr_ml.data import IBDataError

        trader = make_trader()
        error = IBDataError("boom", ib_errors=[{"code": 200, "message": "No security definition"}])
        assert trader._data_error_reason(error) == "data_error:IBKR_200"

    def test_a_code_is_not_matched_by_prefix(self, make_trader):
        from ibkr_ml.data import IBDataError

        trader = make_trader()
        error = IBDataError("boom", ib_errors=[{"code": 1620, "message": "unrelated"}])
        assert trader._data_error_reason(error) == "data_error:IBKR_1620"

    def test_an_error_without_codes_falls_back_to_the_type(self, make_trader):
        trader = make_trader()
        assert trader._data_error_reason(TimeoutError("slow")) == "data_error:TimeoutError"


class TestOrderAcknowledgement:
    """An order TWS never passed on must not be logged as a trade.

    This is the failure that hid in production: TWS held the order behind its
    precautions dialog, the order sat in PendingSubmit and vanished on
    disconnect, and the loop still wrote order_submitted and spent two slots of
    the daily trade budget on trades that never happened.
    """

    def test_an_acknowledged_order_is_logged_as_submitted(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status="Submitted")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        events = [e["event_type"] for e in read_log_events()]
        assert "order_submitted" in events
        assert "order_not_transmitted" not in events
        assert trader.daily_trade_count == 1

    def test_a_stuck_order_is_logged_as_not_transmitted(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status="PendingSubmit")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        events = [e["event_type"] for e in read_log_events()]
        assert "order_not_transmitted" in events
        assert "order_submitted" not in events

    def test_a_stuck_order_does_not_spend_the_daily_trade_budget(self, make_trader):
        trader = make_trader()
        ib = FakeIB(placed_order_status="PendingSubmit")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        assert trader.daily_trade_count == 0

    def test_a_stuck_order_prints_the_tws_setting_to_check(self, make_trader, capsys):
        trader = make_trader()
        ib = FakeIB(placed_order_status="PendingSubmit")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        printed = capsys.readouterr().out
        assert "Bypass Order Precautions" in printed
        assert "PendingSubmit" in printed

    def test_a_rejected_order_is_recorded_with_its_status(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status="Cancelled")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        submitted = [e for e in read_log_events() if e["event_type"] == "order_submitted"]
        assert submitted[0]["payload"]["order_status"] == "Cancelled"

    def test_a_stuck_bracket_entry_is_also_caught(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status="PendingSubmit")

        trader._submit_target(ib, TradeDecision("AAA", "BUY", 0.9, 0, 100, 50.0, "model_entry"))

        events = [e["event_type"] for e in read_log_events()]
        assert "order_not_transmitted" in events
        # All three bracket legs still went out; it is the parent's fate that
        # decides whether this counts as a trade.
        assert len(ib.placed) == 3

    def test_the_logged_status_is_carried_through(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status="PreSubmitted")

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        submitted = [e for e in read_log_events() if e["event_type"] == "order_submitted"]
        assert submitted[0]["payload"]["order_status"] == "PreSubmitted"

    def test_a_broker_without_order_status_still_works(self, make_trader, read_log_events):
        # Older ib_insync builds and stubs may hand back no trade object.
        trader = make_trader()
        ib = FakeIB()
        ib.placeOrder = lambda contract, order: (ib.placed.append((contract, order)) or None)

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        submitted = [e for e in read_log_events() if e["event_type"] == "order_submitted"]
        assert submitted[0]["payload"]["order_status"] == "unknown"
        assert trader.daily_trade_count == 1

    def test_a_transient_cancel_from_a_tws_preset_is_not_a_rejection(
        self, make_trader, read_log_events
    ):
        """Seen live: Cancelled then Submitted then Filled, all inside a second.

        TWS order presets rewrite an order (here the time-in-force) and bounce
        it through Cancelled before resubmitting. Reading that first status and
        stopping would report a filled order as rejected.
        """
        trader = make_trader()
        ib = FakeIB(placed_order_status=["Cancelled", "Submitted", "Filled"])

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        submitted = [e for e in read_log_events() if e["event_type"] == "order_submitted"]
        assert submitted[0]["payload"]["order_status"] in {"Submitted", "Filled"}
        assert trader.daily_trade_count == 1

    def test_a_cancel_that_persists_is_reported_as_the_final_status(
        self, make_trader, read_log_events
    ):
        trader = make_trader()
        ib = FakeIB(placed_order_status=["Cancelled", "Cancelled"])

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        submitted = [e for e in read_log_events() if e["event_type"] == "order_submitted"]
        assert submitted[0]["payload"]["order_status"] == "Cancelled"

    def test_an_order_that_settles_after_being_held_is_accepted(self, make_trader, read_log_events):
        trader = make_trader()
        ib = FakeIB(placed_order_status=["PendingSubmit", "PendingSubmit", "Submitted"])

        trader._submit_target(ib, TradeDecision("AAA", "SELL", 0.1, 100, 0, 50.0, "model_exit"))

        events = [e["event_type"] for e in read_log_events()]
        assert "order_submitted" in events
        assert "order_not_transmitted" not in events


class TestReferenceFrames:
    """A model trained with cross-asset features cannot be scored without them.

    Filling the missing columns with zeroes would tell the model "the market
    did not move", which is a statement, not a gap. So a missing or stale
    reference has to block scoring - while protective exits keep running,
    because they never consult the model.
    """

    def bundle_with_references(self, references):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["reference_symbols"] = references
        return bundle

    def test_no_references_configured_means_nothing_is_fetched(self, make_trader):
        trader = make_trader()
        frames, reason = trader._fetch_reference_frames(FakeIB(), et("2026-01-05 11:00"))
        assert frames == {}
        assert reason is None

    def test_references_are_fetched_when_the_model_needs_them(self, make_trader, monkeypatch):
        import pandas as pd

        from ibkr_ml import execution

        trader = make_trader(bundle=self.bundle_with_references({"mkt": "SPY"}))
        fetched = []

        def fake_fetch(ib, symbol, **kwargs):
            fetched.append(symbol)
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2026-01-05 10:55")],
                    "open": [100.0], "high": [101.0], "low": [99.0],
                    "close": [100.5], "volume": [1000.0],
                }
            )

        monkeypatch.setattr(execution, "fetch_historical_frame", fake_fetch)
        frames, reason = trader._fetch_reference_frames(FakeIB(), et("2026-01-05 11:00"))

        assert fetched == ["SPY"]
        assert reason is None
        assert set(frames) == {"mkt"}

    def test_a_failed_reference_download_blocks_scoring(self, make_trader, monkeypatch, read_log_events):
        from ibkr_ml import execution

        trader = make_trader(bundle=self.bundle_with_references({"mkt": "SPY"}))

        def failing_fetch(ib, symbol, **kwargs):
            raise TimeoutError("no data")

        monkeypatch.setattr(execution, "fetch_historical_frame", failing_fetch)
        frames, reason = trader._fetch_reference_frames(FakeIB(), et("2026-01-05 11:00"))

        assert frames is None
        assert reason.startswith("reference_mkt_")
        assert any(e["event_type"] == "reference_error" for e in read_log_events())

    def test_a_stale_reference_blocks_scoring(self, make_trader, monkeypatch, read_log_events):
        import pandas as pd

        from ibkr_ml import execution

        trader = make_trader(bundle=self.bundle_with_references({"mkt": "SPY"}))

        def stale_fetch(ib, symbol, **kwargs):
            return pd.DataFrame(
                {
                    "timestamp": [pd.Timestamp("2026-01-05 09:40")],  # an hour behind
                    "open": [100.0], "high": [101.0], "low": [99.0],
                    "close": [100.5], "volume": [1000.0],
                }
            )

        monkeypatch.setattr(execution, "fetch_historical_frame", stale_fetch)
        frames, reason = trader._fetch_reference_frames(FakeIB(), et("2026-01-05 11:00"))

        assert frames is None
        assert reason == "reference_mkt_stale_data"
        blocked = [e for e in read_log_events() if e["event_type"] == "reference_blocked"]
        assert blocked[0]["payload"]["symbol"] == "SPY"

    def test_the_first_broken_reference_stops_the_rest(self, make_trader, monkeypatch):
        from ibkr_ml import execution

        trader = make_trader(
            bundle=self.bundle_with_references({"mkt": "SPY", "sector": "XLK"})
        )
        attempted = []

        def failing_fetch(ib, symbol, **kwargs):
            attempted.append(symbol)
            raise TimeoutError("no data")

        monkeypatch.setattr(execution, "fetch_historical_frame", failing_fetch)
        trader._fetch_reference_frames(FakeIB(), et("2026-01-05 11:00"))
        # No point downloading the rest once the feature set is already incomplete.
        assert attempted == ["SPY"]


class TestThresholdQualityGate:
    """A model whose best threshold barely trades must not reach the account."""

    def test_an_unqualified_threshold_is_refused(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["threshold_qualified"] = False
        bundle["threshold_note"] = "only 14 trades"

        with pytest.raises(RuntimeError, match="threshold_not_qualified"):
            make_trader(bundle=bundle)

    def test_the_note_explains_why(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["threshold_qualified"] = False
        bundle["threshold_note"] = "exposure 0.1% below 5%"

        with pytest.raises(RuntimeError, match="exposure 0.1%"):
            make_trader(bundle=bundle)

    def test_walk_forward_folds_that_could_not_be_judged_are_refused(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["walk_forward_summary"]["qualified_folds"] = 1  # of 3

        with pytest.raises(RuntimeError, match="walk_forward_qualified_folds=1/3"):
            make_trader(bundle=bundle)

    def test_a_fully_qualified_model_still_passes(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle["threshold_qualified"] = True
        bundle["walk_forward_summary"]["qualified_folds"] = 3

        assert make_trader(bundle=bundle) is not None

    def test_older_bundles_without_the_field_are_not_penalised(self, make_trader):
        from conftest import passing_bundle

        bundle = passing_bundle()
        bundle.pop("threshold_qualified", None)
        bundle["walk_forward_summary"].pop("qualified_folds", None)

        assert make_trader(bundle=bundle) is not None

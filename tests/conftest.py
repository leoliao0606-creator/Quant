"""Shared fixtures for the paper-trading tests.

Nothing here touches IBKR. The fake broker below records what was sent to it,
which is the only way to assert on order placement without a live TWS session.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ibkr_ml.config import MarketDataConfig, ModelConfig, RiskConfig  # noqa: E402


@dataclass
class FakeAccountValue:
    tag: str
    value: str
    currency: str = "USD"


@dataclass
class FakeContract:
    symbol: str
    secType: str = "STK"


@dataclass
class FakePosition:
    contract: FakeContract
    position: float
    avgCost: float


@dataclass
class FakeOrderStatus:
    status: str = "Submitted"


class SequencedOrderStatus:
    """Reports a scripted sequence of statuses, one per read.

    A TWS order preset can bounce an order through Cancelled and back to
    Submitted inside the same second, so the acknowledgement check has to
    survive reading a transient status. A plain dataclass cannot express that.
    """

    def __init__(self, statuses):
        self._statuses = list(statuses)
        self._index = 0

    @property
    def status(self) -> str:
        value = self._statuses[min(self._index, len(self._statuses) - 1)]
        self._index += 1
        return value


@dataclass
class FakeOrder:
    orderId: int
    orderType: str
    action: str = "SELL"
    totalQuantity: float = 0.0


@dataclass
class FakeTrade:
    contract: FakeContract
    order: FakeOrder
    orderStatus: object = field(default_factory=FakeOrderStatus)


class FakeClient:
    def __init__(self) -> None:
        self.next_id = 1000

    def getReqId(self) -> int:
        self.next_id += 1
        return self.next_id


class FakeIB:
    """Records placed and cancelled orders instead of talking to TWS."""

    def __init__(
        self,
        net_liquidation=100000.0,
        positions=(),
        open_trades=(),
        placed_order_status="Submitted",
    ):
        self.net_liquidation = net_liquidation
        self._positions = list(positions)
        self._open_trades = list(open_trades)
        # Status the broker reports for orders placed through it. "Submitted"
        # is the healthy path; "PendingSubmit" reproduces TWS holding an order
        # behind its precautions dialog.
        self.placed_order_status = placed_order_status
        self.client = FakeClient()
        self.placed: list[tuple[FakeContract, object]] = []
        self.cancelled: list[FakeOrder] = []
        self.qualified: list[FakeContract] = []
        self.slept = 0.0
        self.cancel_should_raise = False

    def accountSummary(self):
        return [FakeAccountValue("NetLiquidation", str(self.net_liquidation))]

    def positions(self):
        return self._positions

    def openTrades(self):
        return self._open_trades

    def qualifyContracts(self, contract):
        self.qualified.append(contract)
        return [contract]

    def placeOrder(self, contract, order):
        self.placed.append((contract, order))
        if isinstance(self.placed_order_status, (list, tuple)):
            order_status = SequencedOrderStatus(self.placed_order_status)
        else:
            order_status = FakeOrderStatus(self.placed_order_status)
        return FakeTrade(
            contract,
            FakeOrder(getattr(order, "orderId", 0), order.orderType),
            order_status,
        )

    def cancelOrder(self, order):
        if self.cancel_should_raise:
            raise RuntimeError("cancel rejected by TWS")
        self.cancelled.append(order)

    def sleep(self, seconds):
        self.slept += seconds

    def disconnect(self):
        pass


def passing_bundle(feature_columns=None):
    """A model bundle that clears every deployment gate in _validate_model_bundle."""
    return {
        "model": None,
        "feature_columns": list(feature_columns or ["ret_1"]),
        "model_config": {"max_active_positions": 2},
        "thresholds": {"entry_probability": 0.60, "exit_probability": 0.50},
        "test_metrics": {"auc": 0.65},
        "walk_forward_summary": {
            "fold_count": 3,
            "profitable_folds": 3,
            "mean_sharpe": 1.20,
            "worst_max_drawdown": -0.03,
        },
    }


@pytest.fixture
def bundle_path(tmp_path):
    """Write a passing bundle to disk and hand back its path."""
    import joblib

    path = tmp_path / "model.joblib"
    joblib.dump(passing_bundle(), path)
    return path


@pytest.fixture
def make_trader(tmp_path, bundle_path):
    """Build an IBKRPaperTrader with test configs and a temporary log directory."""
    from ibkr_ml.execution import IBKRPaperTrader

    def build(*, market_config=None, model_config=None, risk_config=None, bundle=None):
        if bundle is not None:
            import joblib

            joblib.dump(bundle, bundle_path)

        model_config = model_config or ModelConfig(model_path=bundle_path)
        model_config.model_path = bundle_path
        risk_config = risk_config or RiskConfig()
        risk_config.log_dir = tmp_path / "logs"
        trader = IBKRPaperTrader(
            connection_config=None,
            market_config=market_config or MarketDataConfig(symbols=("AAA", "BBB")),
            model_config=model_config,
            risk_config=risk_config,
        )
        # Keep the acknowledgement wait short: the tests drive a fake broker
        # whose sleep does not advance the clock, so the real default would
        # spin for five wall-clock seconds per stuck order.
        trader.order_acknowledgement_timeout = 0.05
        return trader

    return build


@pytest.fixture
def read_log_events(tmp_path):
    """Read back every JSONL event the trader wrote during a test."""
    import json

    def read():
        events = []
        for path in sorted((tmp_path / "logs").glob("*.jsonl")):
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    events.append(json.loads(line))
        return events

    return read

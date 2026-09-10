from __future__ import annotations

import time
from typing import Any, NamedTuple


# IBKR pushes purely informational status notices through the same errorEvent
# as real errors: 2104 "Market data farm connection is OK", 2106 "HMDS data
# farm connection is OK", 2158 "Sec-def data farm connection is OK". They carry
# no contract, so a symbol filter alone lets them through.
IB_INFORMATIONAL_CODE_RANGE = (2100, 2200)

# Failures where retrying with a smaller chunk cannot help, because the problem
# is the connection or the session rather than the size of the request.
IB_NON_RETRYABLE_CODES = frozenset({326, 502, 504, 1100, 1300})


def _is_informational_code(code: Any) -> bool:
    try:
        numeric = int(code)
    except (TypeError, ValueError):
        return False
    low, high = IB_INFORMATIONAL_CODE_RANGE
    return low <= numeric < high


class IBDataError(RuntimeError):
    """A historical data failure that keeps IBKR's structured error codes.

    ``codes`` lets callers branch on the actual IBKR error code instead of
    pattern matching a human readable message, and ``original_type`` preserves
    the class name of the wrapped exception so a caller's reason string stays
    stable whether or not any IBKR message happened to arrive.
    """

    def __init__(
        self,
        message: str,
        ib_errors: Any = (),
        original_type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.ib_errors = [dict(error) for error in ib_errors]
        self.original_type = original_type or "RuntimeError"

    @property
    def codes(self) -> list[int]:
        numeric_codes = []
        for error in self.ib_errors:
            try:
                numeric_codes.append(int(error.get("code")))
            except (TypeError, ValueError):
                continue
        return numeric_codes

    @property
    def ib_messages(self) -> list[str]:
        return [str(error.get("message", "")) for error in self.ib_errors]

    @property
    def is_session_conflict(self) -> bool:
        """True for IBKR 162 raised because another IP holds the TWS session.

        Code 162 covers both a benign "HMDS query returned no data" answer and
        a session already connected elsewhere; only the latter is hopeless.
        """
        for error in self.ib_errors:
            try:
                code = int(error.get("code"))
            except (TypeError, ValueError):
                continue
            if code == 162 and "different ip address" in str(error.get("message", "")).lower():
                return True
        return False

    @property
    def is_retryable(self) -> bool:
        if self.is_session_conflict:
            return False

        codes = self.codes
        # IBKR reports a network blip as 1100 (connectivity lost) followed
        # seconds later by 1102 (restored, data maintained). Seeing both means
        # the link is already back by the time this is read, so the request is
        # worth repeating - treating it as fatal aborted whole downloads over a
        # hiccup that had already fixed itself.
        if 1100 in codes and 1102 in codes:
            return True
        return not any(code in IB_NON_RETRYABLE_CODES for code in codes)


def _missing_dependency(package: str) -> RuntimeError:
    return RuntimeError(
        f"Missing dependency '{package}'. Install packages from requirements.txt first."
    )


def _load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise _missing_dependency("pandas") from exc
    return pd


class IBComponents(NamedTuple):
    """The ib_insync names this project uses, imported lazily.

    A named tuple rather than a bare tuple: every caller used to unpack by
    position, so adding an order type meant editing each unpack site, and a
    mis-ordered unpack bound the wrong class silently instead of failing.
    """

    IB: Any
    MarketOrder: Any
    LimitOrder: Any
    StopOrder: Any
    Stock: Any
    util: Any


def load_ib_components() -> IBComponents:
    try:
        from ib_insync import IB, LimitOrder, MarketOrder, Stock, StopOrder, util
    except ModuleNotFoundError as exc:
        raise _missing_dependency("ib-insync") from exc
    return IBComponents(
        IB=IB,
        MarketOrder=MarketOrder,
        LimitOrder=LimitOrder,
        StopOrder=StopOrder,
        Stock=Stock,
        util=util,
    )


# Statuses that mean IBKR has the order and is working it. Anything in
# IB_HELD_IN_TWS_STATES is still sitting inside TWS and has not reached IBKR:
# most often because TWS is holding it behind the order precautions dialog,
# waiting for a human to click through it. Such an order disappears when the
# API client disconnects, so treating it as submitted makes the log claim a
# trade that never happened.
IB_ACKNOWLEDGED_STATES = frozenset({"PreSubmitted", "Submitted", "Filled"})
IB_HELD_IN_TWS_STATES = frozenset({"PendingSubmit", "ApiPending"})


def load_order_status_states() -> frozenset[str]:
    """Order statuses that mean the order is still working at IBKR.

    Read from ib_insync so the set cannot drift from the library's own
    definition, with a literal fallback for the case where the attribute is
    missing or renamed.
    """
    try:
        from ib_insync import OrderStatus
    except ModuleNotFoundError as exc:
        raise _missing_dependency("ib-insync") from exc

    active_states = getattr(OrderStatus, "ActiveStates", None)
    if not active_states:
        return frozenset({"ApiPending", "PendingSubmit", "PreSubmitted", "Submitted"})
    return frozenset(str(state) for state in active_states)


def _format_ib_errors(errors: list[dict[str, Any]]) -> str:
    seen = set()
    formatted = []
    for error in errors:
        key = (
            error.get("req_id"),
            error.get("code"),
            error.get("message"),
        )
        if key in seen:
            continue
        seen.add(key)

        req_id = error.get("req_id")
        code = error.get("code")
        message = error.get("message")
        if req_id is None:
            formatted.append(f"code {code}: {message}")
        else:
            formatted.append(f"reqId {req_id} code {code}: {message}")
    return "; ".join(formatted)


def _ib_error_handler(symbol: str, errors: list[dict[str, Any]]):
    def on_error(*args):
        req_id = args[0] if len(args) > 0 else None
        code = args[1] if len(args) > 1 else None
        message = args[2] if len(args) > 2 else ""
        contract = args[3] if len(args) > 3 else None

        # Status notices are not failures. Collecting them would attach
        # unrelated text to every error message and would flip the exception
        # type in _request_historical_frame.
        if _is_informational_code(code):
            return

        contract_symbol = getattr(contract, "symbol", None)
        if contract_symbol and contract_symbol != symbol:
            return

        errors.append(
            {
                "req_id": req_id,
                "code": code,
                "message": message,
            }
        )

    return on_error


def _attach_ib_error_listener(ib: Any, handler: Any) -> bool:
    """Attach handler to ib.errorEvent and report whether it really attached.

    Prefers eventkit's explicit connect(). The ``+=`` form expands to
    ``ib.errorEvent = ib.errorEvent.__iadd__(handler)``, which connects the
    listener *before* the attribute assignment; if that assignment raises, the
    listener is already live while the caller believes it never attached, and
    it then never gets removed.
    """
    event = getattr(ib, "errorEvent", None)
    if event is None:
        return False

    connect = getattr(event, "connect", None)
    if callable(connect):
        connect(handler)
        return True

    try:
        ib.errorEvent += handler
    except AttributeError:
        return False
    return True


def _detach_ib_error_listener(ib: Any, handler: Any, symbol: str) -> None:
    """Remove handler from ib.errorEvent, reporting failures instead of hiding them.

    A listener left attached to a long lived connection keeps appending to a
    list nobody reads any more, on every later IBKR message, so a failure here
    must be visible.
    """
    event = getattr(ib, "errorEvent", None)
    if event is None:
        return

    disconnect = getattr(event, "disconnect", None)
    try:
        if callable(disconnect):
            disconnect(handler)
        else:
            ib.errorEvent -= handler
    except Exception as exc:
        print(
            f"warning: failed to detach the IBKR error listener for {symbol}: "
            f"{exc.__class__.__name__}: {exc}"
        )


def connect_ib(config: Any):
    IB = load_ib_components().IB
    request_timeout = float(getattr(config, "request_timeout", 120.0))
    connect_retries = max(int(getattr(config, "connect_retries", 1)), 1)
    retry_delay_seconds = float(getattr(config, "retry_delay_seconds", 3.0))
    client_id_step = max(int(getattr(config, "client_id_step", 1)), 1)

    last_error = None
    for attempt in range(connect_retries):
        client_id = int(config.client_id) + attempt * client_id_step
        ib = IB()
        ib.RequestTimeout = request_timeout
        try:
            ib.connect(config.host, config.port, clientId=client_id)
            return ib
        except Exception as exc:
            last_error = exc
            try:
                ib.disconnect()
            except Exception:
                pass

            if attempt + 1 >= connect_retries:
                break

            print(
                f"IBKR connect attempt {attempt + 1}/{connect_retries} failed "
                f"for clientId={client_id}: {exc.__class__.__name__}: {exc}"
            )
            time.sleep(retry_delay_seconds)

    raise RuntimeError(
        "Failed to connect to TWS/IB Gateway after "
        f"{connect_retries} attempts starting from clientId={config.client_id}. "
        f"Last error: {last_error}. "
        "Check that TWS is logged in, API access is enabled, the socket port is correct, "
        "and no API permission dialog is waiting for input."
    )


def _parse_duration(duration: str) -> tuple[int, str]:
    parts = duration.strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unsupported duration format: {duration!r}")

    value_text, unit = parts
    return int(value_text), unit.upper()


def _duration_to_days(duration: str) -> int:
    value, unit = _parse_duration(duration)
    unit_to_days = {
        "S": 1 / 86400,
        "D": 1,
        "W": 7,
        "M": 30,
        "Y": 365,
    }
    if unit not in unit_to_days:
        raise ValueError(f"Unsupported duration unit: {unit!r}")
    return max(int(value * unit_to_days[unit]), 1)


# Minutes past midnight of the US equities open. Hourly bars are aligned to it
# so a bar covers 09:30-10:30 rather than 09:00-10:00, which would put the
# opening auction in a bucket that is mostly outside the session.
SESSION_OPEN_OFFSET = "30min"


def bar_size_to_pandas_rule(bar_size: str) -> str:
    """Translate an IBKR bar size such as "1 hour" into a pandas resample rule."""
    parts = str(bar_size).strip().split()
    if len(parts) != 2:
        raise ValueError(f"Unsupported bar size: {bar_size!r}")

    value, unit = parts
    unit = unit.lower().rstrip("s")
    suffix = {"sec": "s", "second": "s", "min": "min", "minute": "min", "hour": "h", "day": "D"}
    if unit not in suffix:
        raise ValueError(f"Unsupported bar size unit: {unit!r}")
    return f"{int(value)}{suffix[unit]}"


def resample_frame(frame, bar_size: str, bar_timezone: str | None = None):
    """Aggregate OHLCV bars up to a longer bar size, in US Eastern time.

    Lets one download serve several timescales, which matters twice over: a
    fresh pull of hourly bars costs another session-holding download, and
    experiments across timescales are only comparable when they are built from
    the same underlying prices.

    Grouping happens on the US Eastern wall clock, not on whatever zone the
    bars arrived in. It has to: a daily bucket in UTC starts at 20:00 the
    previous Eastern evening and would split each session across two rows.

    Intraday buckets are offset to the 09:30 open, so an hourly bar covers
    09:30-10:30 rather than 09:00-10:00 - the latter would file the opening
    auction under a bucket that is mostly outside the session.

    Only aggregation upward is possible. Buckets with no trading - a holiday, a
    half day, the gap around a halt - come out empty and are dropped rather
    than forward-filled: inventing a bar that never traded would hand the model
    a price nobody could have acted on.
    """
    from .features import to_eastern_naive

    rule = bar_size_to_pandas_rule(bar_size)
    eastern = to_eastern_naive(frame["timestamp"], bar_timezone)
    indexed = frame.drop(columns=["timestamp"]).set_index(eastern).sort_index()
    indexed.index.name = "timestamp"

    intraday = rule.endswith(("s", "min", "h"))
    aggregated = indexed.resample(
        rule,
        label="left",
        closed="left",
        offset=SESSION_OPEN_OFFSET if intraday else None,
    ).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})

    aggregated = aggregated.dropna(subset=["open", "high", "low", "close"]).reset_index()
    return aggregated[["timestamp", "open", "high", "low", "close", "volume"]]


def _is_intraday_bar_size(bar_size: str) -> bool:
    lowered = bar_size.lower()
    return "sec" in lowered or "min" in lowered or "hour" in lowered


def _recommended_chunk_duration(duration: str, bar_size: str, max_duration_per_request: str | None) -> str:
    if max_duration_per_request:
        return max_duration_per_request

    total_days = _duration_to_days(duration)
    if not _is_intraday_bar_size(bar_size):
        return f"{min(total_days, 180)} D"

    if total_days <= 90:
        return duration
    if total_days <= 180:
        return "45 D"
    return "30 D"


def _normalize_historical_frame(raw_frame, symbol: str):
    pd = _load_pandas()
    if raw_frame is None:
        raise RuntimeError(
            f"IBKR returned an empty dataframe object for {symbol}. "
            "Check the TWS API error log for a timeout or rejected historical data request."
        )
    if raw_frame.empty:
        raise ValueError(f"No historical bars returned for symbol {symbol}.")

    frame = raw_frame.rename(columns={"date": "timestamp"})
    try:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
    except (ValueError, TypeError):
        # A range spanning a daylight-saving change arrives with two different
        # UTC offsets, which has no single naive dtype. UTC does.
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    frame = frame[columns].sort_values("timestamp").reset_index(drop=True)
    return frame


def _request_historical_frame(
    ib: Any,
    contract: Any,
    symbol: str,
    duration: str,
    bar_size: str,
    use_rth: bool,
    end_datetime: Any,
):
    util = load_ib_components().util

    ib_errors: list[dict[str, Any]] = []
    error_handler = _ib_error_handler(symbol, ib_errors)
    attached = _attach_ib_error_listener(ib, error_handler)

    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime=end_datetime,
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow="TRADES",
            useRTH=use_rth,
            formatDate=1,
            keepUpToDate=False,
        )
    finally:
        if attached:
            _detach_ib_error_listener(ib, error_handler, symbol)

    error_detail = _format_ib_errors(ib_errors)
    suffix = f" IBKR API errors: {error_detail}" if error_detail else ""
    if bars is None:
        raise IBDataError(
            f"IBKR returned no data object for {symbol}. "
            "This usually means the historical data request timed out or was rejected by TWS/Gateway."
            f"{suffix}",
            ib_errors=ib_errors,
            original_type="RuntimeError",
        )

    try:
        return _normalize_historical_frame(util.df(bars), symbol)
    except (RuntimeError, ValueError) as exc:
        # Wrap unconditionally and keep the original class name. Wrapping only
        # when an IBKR message arrived made the exception type - and therefore
        # the caller's reason string - depend on unrelated API chatter.
        raise IBDataError(
            f"{exc}{suffix}",
            ib_errors=ib_errors,
            original_type=exc.__class__.__name__,
        ) from exc


def _fetch_chunked_historical_frame(
    ib: Any,
    contract: Any,
    symbol: str,
    duration: str,
    bar_size: str,
    use_rth: bool,
    max_duration_per_request: str | None,
):
    pd = _load_pandas()

    total_days = _duration_to_days(duration)
    chunk_duration = _recommended_chunk_duration(duration, bar_size, max_duration_per_request)
    chunk_days = _duration_to_days(chunk_duration)
    if chunk_days >= total_days:
        return _request_historical_frame(
            ib=ib,
            contract=contract,
            symbol=symbol,
            duration=duration,
            bar_size=bar_size,
            use_rth=use_rth,
            end_datetime="",
        )

    frames = []
    remaining_days = total_days
    next_end = ""
    previous_earliest = None
    minimum_chunk_days = 7 if _is_intraday_bar_size(bar_size) else 30
    chunk_failures: list[str] = []

    while remaining_days > 0:
        current_chunk_days = min(chunk_days, remaining_days)
        try:
            frame = _request_historical_frame(
                ib=ib,
                contract=contract,
                symbol=symbol,
                duration=f"{current_chunk_days} D",
                bar_size=bar_size,
                use_rth=use_rth,
                end_datetime=next_end,
            )
        except Exception as exc:
            # Keep every attempt's diagnostics. Discarding them lost the IBKR
            # error codes that explain why the download failed.
            failure = f"{current_chunk_days} D chunk: {exc.__class__.__name__}: {exc}"
            chunk_failures.append(failure)
            print(f"warning: historical data for {symbol} failed, {failure}")

            if not getattr(exc, "is_retryable", True):
                # A session conflict or a lost connection fails identically at
                # every chunk size, so halving only burns round trips.
                raise
            if current_chunk_days <= minimum_chunk_days:
                raise IBDataError(
                    f"Historical data for {symbol} failed at the minimum chunk size "
                    f"of {minimum_chunk_days} D. Attempts: " + " | ".join(chunk_failures),
                    ib_errors=getattr(exc, "ib_errors", ()),
                    original_type=getattr(exc, "original_type", exc.__class__.__name__),
                ) from exc
            chunk_days = max(current_chunk_days // 2, minimum_chunk_days)
            continue

        frames.append(frame)
        earliest_timestamp = frame["timestamp"].min()
        if previous_earliest is not None and earliest_timestamp >= previous_earliest:
            break

        previous_earliest = earliest_timestamp
        next_end = (earliest_timestamp - pd.Timedelta(seconds=1)).to_pydatetime()
        remaining_days -= current_chunk_days
        ib.sleep(0.2)

    if not frames:
        detail = " Attempts: " + " | ".join(chunk_failures) if chunk_failures else ""
        raise IBDataError(
            f"No historical chunks were returned for {symbol}.{detail}",
            original_type="RuntimeError",
        )

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return combined


def fetch_historical_frame(
    ib: Any,
    symbol: str,
    duration: str,
    bar_size: str,
    use_rth: bool,
    max_duration_per_request: str | None = None,
):
    Stock = load_ib_components().Stock

    contract = Stock(symbol, "SMART", "USD")
    ib.qualifyContracts(contract)
    return _fetch_chunked_historical_frame(
        ib=ib,
        contract=contract,
        symbol=symbol,
        duration=duration,
        bar_size=bar_size,
        use_rth=use_rth,
        max_duration_per_request=max_duration_per_request,
    )

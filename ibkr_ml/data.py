from __future__ import annotations

import time
from typing import Any


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


def load_ib_components():
    try:
        from ib_insync import IB, MarketOrder, Stock, util
    except ModuleNotFoundError as exc:
        raise _missing_dependency("ib-insync") from exc
    return IB, MarketOrder, Stock, util


def connect_ib(config: Any):
    IB, _, _, _ = load_ib_components()
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
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=False)
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
    _, _, _, util = load_ib_components()

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
    if bars is None:
        raise RuntimeError(
            f"IBKR returned no data object for {symbol}. "
            "This usually means the historical data request timed out or was rejected by TWS/Gateway."
        )

    return _normalize_historical_frame(util.df(bars), symbol)


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
        except Exception:
            if current_chunk_days <= minimum_chunk_days:
                raise
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
        raise RuntimeError(f"No historical chunks were returned for {symbol}.")

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
    _, _, Stock, _ = load_ib_components()

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

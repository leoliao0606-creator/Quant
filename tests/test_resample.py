"""Aggregating bars up to a longer timescale.

One download then serves several timescales, and - more importantly -
experiments across timescales become comparable, because they are built from
the same underlying prices rather than from two separate pulls.
"""

from __future__ import annotations

import pandas as pd
import pytest

from ibkr_ml.data import bar_size_to_pandas_rule, resample_frame


def five_minute_frame(days=2, bars_per_day=78, start="2026-01-05 09:30"):
    """A session-shaped series: 78 five-minute bars per day, 09:30 to 15:55."""
    rows = []
    price = 100.0
    for day in range(days):
        day_start = pd.Timestamp(start) + pd.Timedelta(days=day)
        for bar in range(bars_per_day):
            price += 0.01
            rows.append(
                {
                    "timestamp": day_start + pd.Timedelta(minutes=5 * bar),
                    "open": price,
                    "high": price + 0.5,
                    "low": price - 0.5,
                    "close": price + 0.1,
                    "volume": 100.0,
                }
            )
    return pd.DataFrame(rows)


class TestBarSizeRule:
    def test_known_sizes_translate(self):
        assert bar_size_to_pandas_rule("5 mins") == "5min"
        assert bar_size_to_pandas_rule("30 mins") == "30min"
        assert bar_size_to_pandas_rule("1 hour") == "1h"
        assert bar_size_to_pandas_rule("2 hours") == "2h"
        assert bar_size_to_pandas_rule("1 day") == "1D"

    def test_unsupported_input_is_rejected(self):
        with pytest.raises(ValueError, match="Unsupported bar size"):
            bar_size_to_pandas_rule("hourly")
        with pytest.raises(ValueError, match="Unsupported bar size unit"):
            bar_size_to_pandas_rule("1 fortnight")


class TestAggregation:
    def test_ohlcv_is_aggregated_the_right_way_round(self):
        frame = five_minute_frame(days=1)
        hourly = resample_frame(frame, "1 hour")

        first_bucket = frame[frame["timestamp"] < pd.Timestamp("2026-01-05 10:30")]
        row = hourly.iloc[0]
        assert row["open"] == first_bucket["open"].iloc[0]
        assert row["close"] == first_bucket["close"].iloc[-1]
        assert row["high"] == first_bucket["high"].max()
        assert row["low"] == first_bucket["low"].min()
        assert row["volume"] == first_bucket["volume"].sum()

    def test_hourly_buckets_start_at_the_open(self):
        """09:30-10:30, not 09:00-10:00.

        The latter would file the opening auction under a bucket that is mostly
        outside the session.
        """
        hourly = resample_frame(five_minute_frame(days=1), "1 hour")
        clocks = list(hourly["timestamp"].dt.strftime("%H:%M"))
        assert clocks == ["09:30", "10:30", "11:30", "12:30", "13:30", "14:30", "15:30"]

    def test_a_full_day_of_five_minute_bars_becomes_seven_hourly_bars(self):
        assert len(resample_frame(five_minute_frame(days=1), "1 hour")) == 7
        assert len(resample_frame(five_minute_frame(days=3), "1 hour")) == 21

    def test_thirty_minute_buckets_also_align_to_the_open(self):
        half_hourly = resample_frame(five_minute_frame(days=1), "30 mins")
        assert half_hourly["timestamp"].iloc[0].strftime("%H:%M") == "09:30"
        assert len(half_hourly) == 13  # 09:30..15:30, the last one half length

    def test_empty_buckets_are_dropped_not_invented(self):
        """A holiday must not become a fabricated bar.

        Forward-filling would hand the model a price nobody could have traded.
        """
        frame = five_minute_frame(days=1)
        gapped = pd.concat(
            [frame, five_minute_frame(days=1, start="2026-01-09 09:30")], ignore_index=True
        )
        hourly = resample_frame(gapped, "1 hour")
        # Two trading days of seven bars each, and nothing for the days between.
        assert len(hourly) == 14
        assert set(hourly["timestamp"].dt.date.astype(str)) == {"2026-01-05", "2026-01-09"}

    def test_daily_buckets_follow_the_eastern_date(self):
        daily = resample_frame(five_minute_frame(days=3), "1 day")
        assert len(daily) == 3
        assert list(daily["timestamp"].dt.strftime("%Y-%m-%d")) == [
            "2026-01-05", "2026-01-06", "2026-01-07"
        ]

    def test_output_is_naive_eastern_whatever_went_in(self):
        frame = five_minute_frame(days=1)
        as_utc = frame.copy()
        as_utc["timestamp"] = (
            as_utc["timestamp"].dt.tz_localize("America/New_York").dt.tz_convert("UTC")
        )

        from_naive = resample_frame(frame, "1 hour")
        from_utc = resample_frame(as_utc, "1 hour")

        assert from_utc["timestamp"].dt.tz is None
        pd.testing.assert_frame_equal(from_naive, from_utc)

    def test_a_declared_source_zone_is_honoured(self):
        frame = five_minute_frame(days=1)
        shifted = frame.copy()
        shifted["timestamp"] = shifted["timestamp"] + pd.Timedelta(hours=5)  # a UTC host

        hourly = resample_frame(shifted, "1 hour", bar_timezone="UTC")
        assert hourly["timestamp"].iloc[0].strftime("%H:%M") == "09:30"

    def test_columns_and_order_are_preserved(self):
        hourly = resample_frame(five_minute_frame(days=1), "1 hour")
        assert list(hourly.columns) == ["timestamp", "open", "high", "low", "close", "volume"]
        assert hourly["timestamp"].is_monotonic_increasing

    def test_aggregated_bars_stay_internally_consistent(self):
        hourly = resample_frame(five_minute_frame(days=3), "1 hour")
        assert (hourly["high"] >= hourly[["open", "close"]].max(axis=1)).all()
        assert (hourly["low"] <= hourly[["open", "close"]].min(axis=1)).all()

    def test_resampling_to_the_same_size_is_a_no_op_in_content(self):
        frame = five_minute_frame(days=1)
        same = resample_frame(frame, "5 mins")
        assert len(same) == len(frame)
        assert same["close"].tolist() == frame["close"].tolist()

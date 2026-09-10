"""Feature construction and labelling.

The look-ahead test is the important one here: a feature that quietly reads a
future bar makes every backtest meaningless while every metric still looks fine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ibkr_ml.features import (
    FEATURE_COLUMNS,
    feature_columns,
    build_feature_frame,
    build_labeled_rows,
    build_latest_feature_row,
    to_eastern_naive,
)


def price_frame(n=200, seed=3):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
    spread = np.abs(rng.normal(0, 0.0008, n)) * close
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 09:30", periods=n, freq="5min"),
            "open": close * (1 + rng.normal(0, 0.0002, n)),
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1000, 9000, n).astype(float),
        }
    )


class TestFeatureFrame:
    def test_every_declared_feature_is_produced(self):
        frame = build_feature_frame(price_frame())
        assert set(FEATURE_COLUMNS).issubset(frame.columns)

    def test_features_do_not_read_future_bars(self):
        """Truncating the input must not change any earlier feature value.

        If a feature used a later bar, cutting the data short would change the
        rows before the cut. Checked on the last 20 usable rows of the short
        frame, where a rolling window is fully warmed up.
        """
        full = build_feature_frame(price_frame(n=200))
        truncated = build_feature_frame(price_frame(n=200).iloc[:150].copy())

        compared = truncated.iloc[130:150][FEATURE_COLUMNS]
        reference = full.iloc[130:150][FEATURE_COLUMNS]
        pd.testing.assert_frame_equal(compared, reference, check_exact=False, rtol=1e-9)

    def test_duplicate_timestamps_are_dropped(self):
        frame = price_frame(n=50)
        doubled = pd.concat([frame, frame], ignore_index=True)
        built = build_feature_frame(doubled)
        assert len(built) == 50
        assert built["timestamp"].is_monotonic_increasing

    def test_infinities_are_replaced_with_nan(self):
        frame = price_frame(n=60)
        # A zero-range bar makes close_location divide by zero.
        frame.loc[30, ["high", "low", "close"]] = 100.0
        built = build_feature_frame(frame)
        assert not np.isinf(built[FEATURE_COLUMNS].to_numpy(dtype=float)).any()


class TestLabels:
    def test_label_matches_the_forward_return_threshold(self):
        rows = build_labeled_rows("AAA", price_frame(), horizon_bars=3, positive_return_threshold=0.001)
        expected = (rows["future_return"] > 0.001).astype(int)
        pd.testing.assert_series_equal(rows["target"], expected, check_names=False)

    def test_a_higher_threshold_produces_fewer_positives(self):
        loose = build_labeled_rows("AAA", price_frame(), 3, 0.0000)
        strict = build_labeled_rows("AAA", price_frame(), 3, 0.0050)
        assert strict["target"].mean() < loose["target"].mean()

    def test_labelled_rows_carry_no_missing_values(self):
        rows = build_labeled_rows("AAA", price_frame(), 3, 0.001)
        assert not rows.isna().any().any()

    def test_too_little_history_is_rejected(self):
        with pytest.raises(ValueError, match="Not enough history"):
            build_labeled_rows("AAA", price_frame(n=10), 3, 0.001)

    def test_the_symbol_is_attached_to_every_row(self):
        rows = build_labeled_rows("XYZ", price_frame(), 3, 0.001)
        assert (rows["symbol"] == "XYZ").all()


class TestLatestRow:
    def test_only_the_newest_usable_bar_is_returned(self):
        latest = build_latest_feature_row("AAA", price_frame())
        assert len(latest) == 1

    def test_the_scoring_row_matches_the_training_row_for_the_same_bar(self):
        """The live path and the training path must agree bar for bar.

        build_latest_feature_row is what the paper trader scores on;
        build_labeled_rows is what the model was fitted on. A divergence means
        the model sees different numbers live than it was trained with, and no
        metric anywhere would reveal it.

        The frame is cut at the bar under test so the live path sees exactly
        what it would see at that moment, which also pins down that scoring
        does not depend on bars that had not happened yet.
        """
        frame = price_frame()
        labelled = build_labeled_rows("AAA", frame, horizon_bars=3, positive_return_threshold=0.001)
        target_timestamp = labelled["timestamp"].iloc[-1]

        visible = frame[frame["timestamp"] <= target_timestamp].copy()
        latest = build_latest_feature_row("AAA", visible)
        assert latest["timestamp"].iloc[0] == target_timestamp

        training_row = labelled[labelled["timestamp"] == target_timestamp]
        for column in FEATURE_COLUMNS:
            assert training_row[column].iloc[0] == pytest.approx(latest[column].iloc[0]), column

    def test_too_little_history_is_rejected(self):
        with pytest.raises(ValueError, match="Not enough history"):
            build_latest_feature_row("AAA", price_frame(n=5))


class TestTimezoneNormalization:
    """The same bar must produce the same features wherever TWS runs.

    TWS returns naive timestamps in its own machine's wall clock. Before this
    was handled, a model trained on a US Eastern host and deployed on a UTC one
    read a five-hour-shifted time of day, which no metric would have surfaced.
    """

    def eastern_and_utc_frames(self, n=120):
        eastern = price_frame(n=n)
        utc = eastern.copy()
        # The identical bars as a UTC host would stamp them: January is UTC-5.
        utc["timestamp"] = utc["timestamp"] + pd.Timedelta(hours=5)
        return eastern, utc

    def test_utc_bars_declared_as_utc_match_eastern_bars(self):
        eastern, utc = self.eastern_and_utc_frames()
        from_eastern = build_feature_frame(eastern)
        from_utc = build_feature_frame(utc, bar_timezone="UTC")
        pd.testing.assert_frame_equal(
            from_eastern[FEATURE_COLUMNS], from_utc[FEATURE_COLUMNS], check_exact=False, rtol=1e-9
        )

    def test_undeclared_utc_bars_produce_different_time_of_day(self):
        # The bug this guards against: without the declaration the same bar
        # lands at a different point of the session.
        eastern, utc = self.eastern_and_utc_frames()
        from_eastern = build_feature_frame(eastern)
        from_utc_undeclared = build_feature_frame(utc)
        assert not np.allclose(
            from_eastern["tod_sin"].to_numpy(), from_utc_undeclared["tod_sin"].to_numpy()
        )

    def test_price_features_never_depend_on_the_timezone(self):
        eastern, utc = self.eastern_and_utc_frames()
        price_columns = [c for c in FEATURE_COLUMNS if c not in {"tod_sin", "tod_cos"}]
        from_eastern = build_feature_frame(eastern)
        from_utc_undeclared = build_feature_frame(utc)
        pd.testing.assert_frame_equal(
            from_eastern[price_columns],
            from_utc_undeclared[price_columns],
            check_exact=False,
            rtol=1e-9,
        )

    def test_timestamps_come_back_naive_in_eastern(self):
        _, utc = self.eastern_and_utc_frames()
        built = build_feature_frame(utc, bar_timezone="UTC")
        assert built["timestamp"].dt.tz is None
        assert built["timestamp"].iloc[0].hour == 9

    def test_timezone_aware_input_is_converted(self):
        frame = price_frame(n=60)
        frame["timestamp"] = frame["timestamp"].dt.tz_localize("America/New_York")
        built = build_feature_frame(frame)
        assert built["timestamp"].dt.tz is None
        assert built["timestamp"].iloc[0].hour == 9

    def test_the_open_maps_to_the_start_of_the_session_encoding(self):
        built = build_feature_frame(price_frame(n=60))
        # 09:30 is minute zero: sin(0) = 0 and cos(0) = 1.
        assert built["tod_sin"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert built["tod_cos"].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_labelling_and_scoring_accept_the_same_declaration(self):
        _, utc = self.eastern_and_utc_frames(n=200)
        labelled = build_labeled_rows("AAA", utc, 3, 0.001, bar_timezone="UTC")
        target_timestamp = labelled["timestamp"].iloc[-1]

        # Convert the cut point back to the UTC clock the raw frame uses.
        visible = utc[utc["timestamp"] <= target_timestamp + pd.Timedelta(hours=5)].copy()
        latest = build_latest_feature_row("AAA", visible, bar_timezone="UTC")

        assert latest["timestamp"].iloc[0] == target_timestamp
        for column in FEATURE_COLUMNS:
            assert labelled[column].iloc[-1] == pytest.approx(latest[column].iloc[0]), column


def reference_frame(n=200, seed=99, drift=0.0):
    """A second price series to measure the main one against."""
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.0009, n)))
    spread = np.abs(rng.normal(0, 0.0006, n)) * close
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2026-01-05 09:30", periods=n, freq="5min"),
            "open": close,
            "high": close + spread,
            "low": close - spread,
            "close": close,
            "volume": rng.integers(1000, 9000, n).astype(float),
        }
    )


class TestCrossAssetFeatures:
    """Relative performance is where directional information lives.

    The single-asset feature set can only describe a stock's own history, which
    is why a model trained on it alone put 88% of its weight on volatility.
    """

    def test_market_reference_adds_its_columns(self):
        from ibkr_ml.features import MARKET_FEATURE_COLUMNS

        built = build_feature_frame(price_frame(), reference_frames={"mkt": reference_frame()})
        for column in MARKET_FEATURE_COLUMNS:
            assert column in built.columns, column

    def test_no_reference_leaves_the_feature_set_untouched(self):
        from ibkr_ml.features import MARKET_FEATURE_COLUMNS

        built = build_feature_frame(price_frame())
        for column in MARKET_FEATURE_COLUMNS:
            assert column not in built.columns

    def test_each_reference_kind_contributes_its_own_block(self):
        from ibkr_ml.features import (
            MARKET_FEATURE_COLUMNS,
            SECTOR_FEATURE_COLUMNS,
            VOLATILITY_INDEX_FEATURE_COLUMNS,
            cross_asset_feature_columns,
        )

        assert cross_asset_feature_columns({"mkt": 1}) == MARKET_FEATURE_COLUMNS
        assert cross_asset_feature_columns({"sector": 1}) == SECTOR_FEATURE_COLUMNS
        assert cross_asset_feature_columns({"volx": 1}) == VOLATILITY_INDEX_FEATURE_COLUMNS
        assert cross_asset_feature_columns({}) == []
        assert cross_asset_feature_columns(None) == []
        assert cross_asset_feature_columns({"mkt": 1, "sector": 1, "volx": 1}) == [
            *MARKET_FEATURE_COLUMNS, *SECTOR_FEATURE_COLUMNS, *VOLATILITY_INDEX_FEATURE_COLUMNS
        ]

    def test_excess_return_is_the_stock_minus_the_market(self):
        stock, market = price_frame(), reference_frame()
        built = build_feature_frame(stock, reference_frames={"mkt": market})
        expected = built["ret_6"] - built["mkt_ret_6"]
        pd.testing.assert_series_equal(
            built["excess_ret_6"], expected, check_names=False, rtol=1e-12
        )

    def test_a_stock_that_is_the_market_has_zero_excess_and_unit_beta(self):
        # Feeding the same series as its own benchmark pins the arithmetic.
        stock = price_frame()
        built = build_feature_frame(stock, reference_frames={"mkt": stock.copy()})
        usable = built.dropna(subset=["excess_ret_1", "beta_60", "corr_60"])
        assert usable["excess_ret_1"].abs().max() == pytest.approx(0.0, abs=1e-12)
        assert usable["beta_60"].iloc[-1] == pytest.approx(1.0, abs=1e-9)
        assert usable["corr_60"].iloc[-1] == pytest.approx(1.0, abs=1e-9)

    def test_cross_asset_features_do_not_read_the_future(self):
        """Truncating the reference must not change any earlier feature value.

        merge_asof(direction="backward") is what guarantees this; a
        forward-filling join would pair a bar with a reference bar that had not
        happened yet, and every metric downstream would still look fine.
        """
        stock, market = price_frame(n=200), reference_frame(n=200)
        full = build_feature_frame(stock, reference_frames={"mkt": market})
        truncated = build_feature_frame(
            stock.iloc[:150].copy(), reference_frames={"mkt": market.iloc[:150].copy()}
        )

        columns = [c for c in full.columns if c.startswith(("mkt_", "excess_", "beta_", "corr_", "rel_", "vol_ratio_to"))]
        pd.testing.assert_frame_equal(
            truncated.iloc[130:150][columns],
            full.iloc[130:150][columns],
            check_exact=False,
            rtol=1e-9,
        )

    def test_rewriting_future_reference_bars_changes_nothing_earlier(self):
        stock, market = price_frame(n=200), reference_frame(n=200)
        tampered = market.copy()
        tampered.loc[150:, ["open", "high", "low", "close"]] *= 3.0

        original = build_feature_frame(stock, reference_frames={"mkt": market})
        after = build_feature_frame(stock, reference_frames={"mkt": tampered})

        columns = [c for c in original.columns if c.startswith(("mkt_", "excess_", "beta_", "corr_"))]
        pd.testing.assert_frame_equal(
            after.iloc[100:145][columns],
            original.iloc[100:145][columns],
            check_exact=False,
            rtol=1e-9,
        )

    def test_a_gap_in_the_reference_uses_the_last_known_bar(self):
        stock, market = price_frame(n=100), reference_frame(n=100)
        # Drop a stretch of reference bars, as a halt or a thin ETF would.
        gapped = market.drop(market.index[40:50]).reset_index(drop=True)

        built = build_feature_frame(stock, reference_frames={"mkt": gapped})
        assert len(built) == 100
        assert built["mkt_close"].iloc[40:50].notna().all()
        # The carried value is the last bar before the gap, never a later one.
        assert built["mkt_close"].iloc[45] == pytest.approx(market["close"].iloc[39])

    def test_labelled_rows_carry_the_cross_asset_columns(self):
        from ibkr_ml.features import MARKET_FEATURE_COLUMNS

        rows = build_labeled_rows(
            "AAA", price_frame(), 3, 0.001, reference_frames={"mkt": reference_frame()}
        )
        for column in MARKET_FEATURE_COLUMNS:
            assert column in rows.columns
        assert not rows.isna().any().any()

    def test_scoring_and_training_agree_with_references_too(self):
        stock, market = price_frame(n=250), reference_frame(n=250)
        references = {"mkt": market}
        labelled = build_labeled_rows("AAA", stock, 3, 0.001, reference_frames=references)
        target_timestamp = labelled["timestamp"].iloc[-1]

        visible_stock = stock[stock["timestamp"] <= target_timestamp].copy()
        visible_market = market[market["timestamp"] <= target_timestamp].copy()
        latest = build_latest_feature_row(
            "AAA", visible_stock, reference_frames={"mkt": visible_market}
        )

        assert latest["timestamp"].iloc[0] == target_timestamp
        for column in feature_columns(references):
            assert labelled[column].iloc[-1] == pytest.approx(latest[column].iloc[0]), column

    def test_sector_and_volatility_blocks_compute(self):
        built = build_feature_frame(
            price_frame(),
            reference_frames={
                "mkt": reference_frame(seed=1),
                "sector": reference_frame(seed=2),
                "volx": reference_frame(seed=3),
            },
        )
        usable = built.dropna()
        assert len(usable) > 50
        assert usable["sector_excess_ret_6"].notna().all()
        assert usable["volx_z_60"].notna().all()


class TestScoringRefusesIncompleteFeatures:
    def test_predict_refuses_when_a_trained_reference_is_missing(self):
        """Silently zero-filling a missing return would be a wrong statement.

        reindex(fill_value=0.0) turns an absent cross-asset column into "the
        market was flat", which the model reads as real information.
        """
        from ibkr_ml.modeling import predict_probability

        bundle = {
            "feature_columns": ["ret_1", "mkt_ret_1"],
            "base_feature_columns": ["ret_1", "mkt_ret_1"],
            "reference_symbols": {"mkt": "SPY"},
            "model": None,
        }
        with pytest.raises(ValueError, match="mkt_ret_1"):
            predict_probability(bundle, "AAA", price_frame(), reference_frames=None)


class TestLabelModes:
    """The label decides what the model is rewarded for learning."""

    def two_regime_frame(self, n=600, seed=17):
        """Low volatility for the first half, high volatility for the second.

        Under an absolute threshold the high-volatility half produces far more
        positives, because both directions clear a fixed bar more often. That
        is the bias the volatility-scaled label exists to remove.
        """
        rng = np.random.default_rng(seed)
        quiet = rng.normal(0, 0.0004, n // 2)
        wild = rng.normal(0, 0.0035, n - n // 2)
        returns = np.concatenate([quiet, wild])
        close = 100 * np.exp(np.cumsum(returns))
        spread = np.abs(returns) * close
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2026-01-05 09:30", periods=n, freq="5min"),
                "open": close,
                "high": close + spread,
                "low": close - spread,
                "close": close,
                "volume": rng.integers(1000, 9000, n).astype(float),
            }
        )

    def positive_rate_by_regime(self, label_mode, **kwargs):
        rows = build_labeled_rows(
            "AAA", self.two_regime_frame(), horizon_bars=3,
            positive_return_threshold=0.001, label_mode=label_mode, **kwargs
        )
        midpoint = rows["timestamp"].iloc[len(rows) // 2]
        quiet = rows[rows["timestamp"] < midpoint]["target"].mean()
        wild = rows[rows["timestamp"] >= midpoint]["target"].mean()
        return float(quiet), float(wild)

    def test_absolute_labels_are_far_more_common_when_volatility_is_high(self):
        quiet, wild = self.positive_rate_by_regime("absolute")
        # This is the defect: the label itself rewards predicting volatility.
        assert wild > quiet * 3

    def test_volatility_scaling_largely_removes_that_bias(self):
        quiet, wild = self.positive_rate_by_regime(
            "volatility_scaled", volatility_threshold_multiple=0.5
        )
        assert quiet > 0.05 and wild > 0.05
        ratio = wild / quiet
        assert 0.5 < ratio < 2.0, f"positive rate still regime-dependent (ratio {ratio:.2f})"

    def test_direction_labels_are_balanced_in_both_regimes(self):
        quiet, wild = self.positive_rate_by_regime("direction")
        assert 0.35 < quiet < 0.65
        assert 0.35 < wild < 0.65

    def test_direction_mode_is_just_the_sign_of_the_forward_return(self):
        rows = build_labeled_rows("AAA", price_frame(), 3, 0.001, label_mode="direction")
        expected = (rows["future_return"] > 0).astype(int)
        pd.testing.assert_series_equal(rows["target"], expected, check_names=False)

    def test_absolute_mode_still_matches_the_threshold(self):
        rows = build_labeled_rows("AAA", price_frame(), 3, 0.002, label_mode="absolute")
        expected = (rows["future_return"] > 0.002).astype(int)
        pd.testing.assert_series_equal(rows["target"], expected, check_names=False)

    def test_volatility_mode_compares_against_scaled_atr(self):
        rows = build_labeled_rows(
            "AAA", price_frame(), 3, 0.001,
            label_mode="volatility_scaled", volatility_threshold_multiple=0.75,
        )
        expected = (rows["future_return"] > 0.75 * rows["atr_14_pct"]).astype(int)
        pd.testing.assert_series_equal(rows["target"], expected, check_names=False)

    def test_an_unknown_mode_is_rejected(self):
        with pytest.raises(ValueError, match="Unknown label_mode"):
            build_labeled_rows("AAA", price_frame(), 3, 0.001, label_mode="whatever")

    def test_a_bigger_multiple_produces_fewer_positives(self):
        loose = build_labeled_rows(
            "AAA", price_frame(), 3, 0.001,
            label_mode="volatility_scaled", volatility_threshold_multiple=0.2,
        )
        strict = build_labeled_rows(
            "AAA", price_frame(), 3, 0.001,
            label_mode="volatility_scaled", volatility_threshold_multiple=2.0,
        )
        assert strict["target"].mean() < loose["target"].mean()


class TestMixedUtcOffsets:
    """A year of US bars spans a daylight-saving change.

    Inside pandas the series is one tz-aware column and the change is
    invisible. Through any text format the type is lost and what comes back
    mixes -04:00 and -05:00, which pandas either refuses to parse or silently
    returns as object dtype depending on the version. This broke every training
    run that read from the CSV cache.
    """

    def mixed_offset_strings(self):
        return pd.Series(
            [
                "2025-10-31 15:50:00-04:00",
                "2025-10-31 15:55:00-04:00",
                "2025-11-03 09:30:00-05:00",  # first bar after the change
                "2025-11-03 09:35:00-05:00",
            ]
        )

    def test_mixed_offsets_parse_to_one_dtype(self):
        from ibkr_ml.features import parse_bar_timestamps

        parsed = parse_bar_timestamps(self.mixed_offset_strings())
        assert pd.api.types.is_datetime64_any_dtype(parsed)
        assert len(parsed) == 4

    def test_the_wall_clock_survives_the_change(self):
        eastern = to_eastern_naive(self.mixed_offset_strings())
        assert eastern.dt.tz is None
        # Both sides of the change opened at 09:30 local time, and the last two
        # entries must still read 09:30 and 09:35 after conversion.
        assert eastern.iloc[2].strftime("%H:%M") == "09:30"
        assert eastern.iloc[3].strftime("%H:%M") == "09:35"
        assert eastern.iloc[0].strftime("%H:%M") == "15:50"

    def test_ordering_is_preserved_across_the_change(self):
        eastern = to_eastern_naive(self.mixed_offset_strings())
        assert eastern.is_monotonic_increasing

    def test_single_offset_input_is_unaffected(self):
        stamps = pd.Series(["2026-01-05 09:30:00-05:00", "2026-01-05 09:35:00-05:00"])
        eastern = to_eastern_naive(stamps)
        assert eastern.iloc[0].strftime("%H:%M") == "09:30"

    def test_naive_input_still_takes_the_declared_zone(self):
        stamps = pd.Series(["2026-01-05 14:30:00", "2026-01-05 14:35:00"])
        eastern = to_eastern_naive(stamps, bar_timezone="UTC")
        # 14:30 UTC in January is 09:30 US Eastern.
        assert eastern.iloc[0].strftime("%H:%M") == "09:30"

    def test_features_build_from_mixed_offset_bars(self):
        n = 200
        base = pd.date_range("2025-10-30 09:30", periods=n, freq="5min", tz="America/New_York")
        rng = np.random.default_rng(5)
        close = 100 * np.exp(np.cumsum(rng.normal(0, 0.001, n)))
        frame = pd.DataFrame(
            {
                # Round-trip through text, exactly as the CSV cache does.
                "timestamp": pd.Series(base).astype(str),
                "open": close, "high": close * 1.001, "low": close * 0.999,
                "close": close, "volume": rng.integers(1000, 5000, n).astype(float),
            }
        )
        built = build_feature_frame(frame)
        assert len(built) == n
        assert built["timestamp"].dt.tz is None

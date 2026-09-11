#!/usr/bin/env python3
"""Score a trained model on a period no experiment has seen.

Every parameter in this project was chosen by looking at the 360-day window,
so that window can no longer answer whether the result is real - it has been
read too many times. An earlier period can, but only once: the moment a
hold-out is used to choose anything, it stops being a hold-out.

This script therefore refuses to sweep parameters. It takes one configuration,
one model, one period, and prints one answer.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path


def load_holdout_frames(symbols, cache_dir, long_duration, cutoff):
    """Bars strictly before the cutoff, i.e. outside every earlier experiment."""
    import pandas as pd

    from ibkr_ml.cache import load_cached_frame
    from ibkr_ml.features import to_eastern_naive

    frames = {}
    coverage = {}
    for symbol in symbols:
        # The earlier period was fetched under several tags. They do not cover
        # the same span: a "720 D" file ends today, so only its tail falls
        # before the cutoff, while a "540 D" file was fetched ending at the
        # cutoff and covers the whole hold-out. Taking the first tag that
        # exists silently truncated the reference series to 359 trading days,
        # and merge_asof then dropped every traded row before that date.
        # Pick the tag with the most rows before the cutoff instead.
        best = None
        for tag in dict.fromkeys((long_duration, "540 D", "720 D")):
            frame, _ = load_cached_frame(cache_dir, symbol, tag, "5 mins", True)
            if frame is None:
                continue
            stamps = to_eastern_naive(frame["timestamp"])
            keep = stamps < cutoff
            if not keep.any():
                continue
            earlier = frame[keep].copy()
            earlier["timestamp"] = stamps[keep]
            if best is None or len(earlier) > len(best[1]):
                best = (tag, earlier.reset_index(drop=True))
        if best is None or len(best[1]) <= 500:
            continue
        frames[symbol] = best[1]
        coverage[symbol] = (best[0], best[1]["timestamp"].min())

    if coverage:
        # A day or two of difference is just how far back each listing's bars
        # go. Only a gap big enough to cost real hold-out rows is worth saying.
        starts = sorted(t for _, t in coverage.values())
        reference_start = starts[len(starts) // 2]
        tolerance = pd.Timedelta(days=10)
        short = sorted(
            s for s, (_, t) in coverage.items() if t > reference_start + tolerance
        )
        if short:
            latest = max(coverage[s][1] for s in short)
            print(f"提示: {len(short)} 个标的的留出数据比其余晚开始 10 天以上，"
                  f"最晚的从 {latest.date()} 起（多数从 {reference_start.date()} 起）: "
                  f"{short[:8]}{'...' if len(short) > 8 else ''}")
    return frames


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a model on an untouched period.")
    parser.add_argument("--model-path", default="artifacts/exp_Q_gpu.joblib")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--long-duration", default="720 D")
    parser.add_argument(
        "--cutoff",
        default="2025-04-03",
        help="Bars on or after this date were used in earlier experiments and are excluded.",
    )
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument(
        "--dump-trades",
        default=None,
        help="Write the per-trade table here so significance can be tested "
             "afterwards without re-running the hold-out.",
    )
    args = parser.parse_args()

    import pandas as pd

    from ibkr_ml.backtest import simulate_probability_strategy
    from ibkr_ml.config import RiskConfig
    from ibkr_ml.features import build_labeled_rows, feature_columns
    from ibkr_ml.modeling import _encode_features, load_model_bundle

    bundle = load_model_bundle(Path(args.model_path))
    model_config = bundle["model_config"]
    thresholds = bundle["thresholds"]
    references = bundle.get("reference_symbols") or {}
    traded = list(bundle["symbols"])
    needed = sorted(set(traded) | set(references.values()))

    cutoff = pd.Timestamp(args.cutoff)
    frames = load_holdout_frames(needed, args.cache_dir, args.long_duration, cutoff)
    missing = [s for s in needed if s not in frames]
    if missing:
        print(f"缺少留出数据的标的 ({len(missing)}): {missing[:8]}{'...' if len(missing) > 8 else ''}")
    if not frames:
        raise SystemExit("没有可用的留出数据")

    reference_frames = {k: frames[v] for k, v in references.items() if v in frames}
    example = next(iter(frames.values()))
    print(f"留出期: {example['timestamp'].min()} .. {example['timestamp'].max()}")
    print(f"标的 {len([s for s in traded if s in frames])} 个   参照 {list(reference_frames)}")

    rows = []
    for symbol in traded:
        if symbol not in frames:
            continue
        try:
            rows.append(build_labeled_rows(
                symbol=symbol,
                price_frame=frames[symbol],
                horizon_bars=model_config["horizon_bars"],
                positive_return_threshold=model_config["positive_return_threshold"],
                bar_timezone=model_config.get("bar_timezone"),
                reference_frames=reference_frames or None,
                label_mode=model_config.get("label_mode", "absolute"),
                volatility_threshold_multiple=model_config.get("volatility_threshold_multiple", 0.5),
            ))
        except ValueError:
            continue

    dataset = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    columns = bundle.get("base_feature_columns") or feature_columns(reference_frames or None)
    encoded = _encode_features(dataset, columns)
    absent = [c for c in bundle["feature_columns"] if c not in encoded.columns]
    # A missing symbol_* dummy just means that symbol contributed no rows, and
    # filling it with zero is what the model expects. A missing numeric feature
    # is different: reindex would zero it, the model would score a silently
    # crippled matrix, and the low number that came out would say nothing about
    # the strategy. Refuse in that case rather than report a meaningless result.
    absent_numeric = [c for c in absent if not c.startswith("symbol_")]
    if absent_numeric:
        raise SystemExit(
            f"留出集缺少 {len(absent_numeric)} 个训练时用到的数值特征列，"
            f"结果会失真，已中止。前几个: {absent_numeric[:8]}"
        )
    if absent:
        print(f"提示: {len(absent)} 个标的在留出期没有数据，其独热列按 0 填充")
    matrix = encoded.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    dataset["probability_up"] = bundle["model"].predict_proba(matrix)[:, 1]
    print(f"留出样本 {len(dataset):,} 行   {dataset['timestamp'].dt.date.nunique()} 个交易日")

    stored = bundle.get("risk_config") or {}
    known = {f.name for f in fields(RiskConfig)} - {"log_dir"}
    risk_config = replace(RiskConfig(), **{k: v for k, v in stored.items() if k in known})

    result = simulate_probability_strategy(
        prediction_rows=dataset,
        entry_probability=thresholds["entry_probability"],
        exit_probability=thresholds["exit_probability"],
        transaction_cost_bps=args.transaction_cost_bps,
        max_active_positions=model_config.get("max_active_positions", 10),
        risk_config=risk_config,
        probability_ceiling=thresholds.get("probability_ceiling"),
    )

    if args.dump_trades:
        result["trades"].to_csv(args.dump_trades, index=False)
        print(f"每笔交易已写入 {args.dump_trades}（{len(result['trades'])} 行）")

    # The engine annualises the standard deviation of 5-minute returns. Daily
    # returns are the standard non-overlapping unit and, on 540 trading days,
    # a far steadier estimate than the 54-day windows tuning used. Report both
    # rather than replace: the pre-registered criterion names the engine's.
    curve = result["equity_curve"].set_index("timestamp")["equity_curve"]
    daily = curve.resample("1D").last().dropna().pct_change().dropna()
    if len(daily) >= 2 and daily.std(ddof=1) > 0:
        daily_sharpe = float(daily.mean() / daily.std(ddof=1) * (252 ** 0.5))
    else:
        daily_sharpe = float("nan")

    from sklearn.metrics import roc_auc_score

    auc = roc_auc_score(dataset["target"], dataset["probability_up"]) \
        if dataset["target"].nunique() == 2 else float("nan")

    print()
    print("=== 留出期结果（此配置只在此数据上运行这一次）===")
    print(f"  AUC              {auc:.4f}")
    print(f"  交易数           {result['trade_count']}")
    print(f"  峰值总敞口       {result['peak_gross_exposure']:.1%}")
    print(f"  平均总敞口       {result['mean_gross_exposure']:.2%}")
    print(f"  总收益           {result['total_return']:+.2%}")
    print(f"  年化收益         {result['annualized_return']:+.2%}")
    print(f"  Sharpe (逐K线)   {result['sharpe']:.2f}")
    print(f"  Sharpe (按日)    {daily_sharpe:.2f}   ({len(daily)} 个交易日)")
    print(f"  最大回撤         {result['max_drawdown']:+.2%}")
    print()
    passed = (
        result["annualized_return"] > 0.05
        and result["sharpe"] >= 1.0
        and result["max_drawdown"] > -0.15
        and result["trade_count"] >= 100
    )
    print(f"  对照 goal: {'通过' if passed else '未通过'}"
          f"  (年化>5%, Sharpe>=1.0, 回撤>-15%, 交易>=100)")


if __name__ == "__main__":
    main()

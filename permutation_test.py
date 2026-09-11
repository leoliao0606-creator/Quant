#!/usr/bin/env python3
"""Does the model's signal earn the return, or do the trading rules?

A backtest reports what the whole system earned: the model's ranking, the
conviction sizing, the stop and the take-profit, the minimum holding period,
and the plain fact of being long in a market that rose. Only the first of
those is the model, and the others are enough on their own to produce a
respectable-looking curve. One daily-bar fold here scored AUC 0.4969 - worse
than a coin - and still returned +5.56%.

The control is to keep everything and replace only the signal, with a
permutation of itself:

  within   the same probabilities are reassigned among the symbols quoted at
           each timestamp. The multiset at every moment is identical, so the
           same number of names clear the entry threshold at the same times;
           only *which* name gets the high number changes. This isolates
           stock selection.
  global   every probability is reassigned anywhere. Selection and timing
           both go.

If the real result sits inside the permuted distribution, the model did not
earn it.

Read the Sharpe comparison, not the return one. Permuting changes how much
capital is deployed - a real model tends to like the same names for days on
end, so a position is often already open and no new slot is used, while a
scrambled signal picks fresh names daily and fills every slot - and returns
are not comparable across different exposures. Scaling a portfolio toward
cash that earns nothing divides mean and standard deviation by the same
number, so Sharpe is.

Turnover differs too, and turnover costs money. Run once at the real cost
and once at --transaction-cost-bps 0: whatever gap survives at zero is not
commission.
"""

from __future__ import annotations

import argparse
from dataclasses import fields, replace
from pathlib import Path


def load_frames(symbols, cache_dir, duration, bar_size, cutoff, minimum_rows=250):
    """Cached bars for each symbol, truncated to before the cutoff."""
    import pandas as pd

    from ibkr_ml.cache import load_cached_frame
    from ibkr_ml.features import to_eastern_naive

    frames = {}
    for symbol in symbols:
        frame, _ = load_cached_frame(cache_dir, symbol, duration, bar_size, True)
        if frame is None:
            continue
        stamps = to_eastern_naive(frame["timestamp"])
        keep = stamps < pd.Timestamp(cutoff) if cutoff else stamps.notna()
        if keep.sum() < minimum_rows:
            continue
        kept = frame[keep].copy()
        kept["timestamp"] = stamps[keep]
        frames[symbol] = kept.reset_index(drop=True)
    return frames


def score_dataset(bundle, frames):
    """Attach the model's probability to every row it can build features for."""
    import pandas as pd

    from ibkr_ml.features import build_labeled_rows, feature_columns
    from ibkr_ml.modeling import _encode_features

    model_config = bundle["model_config"]
    references = bundle.get("reference_symbols") or {}
    reference_frames = {k: frames[v] for k, v in references.items() if v in frames}

    rows = []
    for symbol in bundle["symbols"]:
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
                volatility_threshold_multiple=model_config.get(
                    "volatility_threshold_multiple", 0.5),
            ))
        except ValueError:
            continue
    if not rows:
        raise SystemExit("没有标的能构建出特征行")

    data = pd.concat(rows, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    columns = bundle.get("base_feature_columns") or feature_columns(reference_frames or None)
    encoded = _encode_features(data, columns)
    absent_numeric = [
        c for c in bundle["feature_columns"]
        if c not in encoded.columns and not c.startswith("symbol_")
    ]
    if absent_numeric:
        raise SystemExit(
            f"缺少 {len(absent_numeric)} 个训练时用到的数值特征列，结果会失真: "
            f"{absent_numeric[:8]}"
        )
    matrix = encoded.reindex(columns=bundle["feature_columns"], fill_value=0.0)
    data["probability_up"] = bundle["model"].predict_proba(matrix)[:, 1]
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Permute the signal, keep everything else.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--start", required=True, help="First bar to score (YYYY-MM-DD).")
    parser.add_argument("--end", required=True, help="Exclusive upper bound (YYYY-MM-DD).")
    parser.add_argument("--cache-dir", default="data_cache")
    parser.add_argument("--duration", default="20 Y")
    parser.add_argument("--bar-size", default="1 day")
    parser.add_argument("--cutoff", default=None,
                        help="Discard bars on or after this date before building features, "
                             "so the run reads only what the experiment was allowed to read.")
    parser.add_argument("--transaction-cost-bps", type=float, default=5.0)
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260910)
    parser.add_argument("--dump", default=None, help="Write every permutation's result here.")
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    from ibkr_ml.backtest import simulate_probability_strategy
    from ibkr_ml.config import RiskConfig
    from ibkr_ml.modeling import load_model_bundle

    bundle = load_model_bundle(Path(args.model_path))
    model_config = bundle["model_config"]
    references = bundle.get("reference_symbols") or {}
    wanted = sorted(set(bundle["symbols"]) | set(references.values()))
    frames = load_frames(wanted, args.cache_dir, args.duration, args.bar_size, args.cutoff)
    data = score_dataset(bundle, frames)

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    data = data[(data["timestamp"] >= start) & (data["timestamp"] < end)].reset_index(drop=True)
    if data.empty:
        raise SystemExit(f"{args.start} .. {args.end} 之间没有可评分的行")

    stored = bundle.get("risk_config") or {}
    known = {f.name for f in fields(RiskConfig)} - {"log_dir"}
    risk_config = replace(RiskConfig(), **{k: v for k, v in stored.items() if k in known})
    thresholds = bundle["thresholds"]

    def run(frame):
        return simulate_probability_strategy(
            prediction_rows=frame,
            entry_probability=thresholds["entry_probability"],
            exit_probability=thresholds["exit_probability"],
            transaction_cost_bps=args.transaction_cost_bps,
            max_active_positions=model_config.get("max_active_positions", 10),
            risk_config=risk_config,
            probability_ceiling=thresholds.get("probability_ceiling"),
        )

    actual = run(data)
    print(f"期间 {data['timestamp'].min().date()} .. {data['timestamp'].max().date()}   "
          f"{len(data):,} 行   {data['symbol'].nunique()} 个标的   "
          f"成本 {args.transaction_cost_bps:g} bp/单边")
    print(f"真实信号: 年化 {actual['annualized_return']:+.2%}   Sharpe {actual['sharpe']:.2f}   "
          f"敞口 {actual['mean_gross_exposure']:.2%}   交易 {actual['trade_count']}")
    print()

    rng = np.random.default_rng(args.seed)
    records = []
    for label, mode in (("同一时刻内打乱（只破坏选股）", "within"),
                        ("完全打乱（选股与择时都破坏）", "global")):
        returns, sharpes, exposures, counts = [], [], [], []
        for _ in range(args.draws):
            shuffled = data.copy()
            if mode == "within":
                shuffled["probability_up"] = (
                    shuffled.groupby("timestamp")["probability_up"]
                    .transform(lambda s: rng.permutation(s.to_numpy())))
            else:
                shuffled["probability_up"] = rng.permutation(
                    shuffled["probability_up"].to_numpy())
            result = run(shuffled)
            returns.append(result["annualized_return"])
            sharpes.append(result["sharpe"])
            exposures.append(result["mean_gross_exposure"])
            counts.append(result["trade_count"])
            records.append({"mode": mode, "annualized_return": result["annualized_return"],
                            "sharpe": result["sharpe"],
                            "exposure": result["mean_gross_exposure"],
                            "trades": result["trade_count"]})
        returns, sharpes = np.array(returns), np.array(sharpes)
        # +1 in numerator and denominator: the observed result is itself one
        # of the arrangements under the null, so a p-value of exactly zero is
        # not available however many draws are taken.
        beat_return = int((returns >= actual["annualized_return"]).sum())
        beat_sharpe = int((sharpes >= actual["sharpe"]).sum())
        print(f"{label}  {args.draws} 次")
        print(f"  年化    中位 {np.median(returns):+.2%}   "
              f"5%~95% [{np.percentile(returns, 5):+.2%}, {np.percentile(returns, 95):+.2%}]   "
              f"p = {(beat_return + 1) / (args.draws + 1):.4f}  "
              f"（{beat_return}/{args.draws} 次不输于真实）")
        print(f"  Sharpe  中位 {np.median(sharpes):.2f}   "
              f"5%~95% [{np.percentile(sharpes, 5):.2f}, {np.percentile(sharpes, 95):.2f}]   "
              f"p = {(beat_sharpe + 1) / (args.draws + 1):.4f}  "
              f"（{beat_sharpe}/{args.draws} 次不输于真实）")
        print(f"  平均敞口 中位 {np.median(exposures):.2%}（真实 {actual['mean_gross_exposure']:.2%}）"
              f"   交易笔数 中位 {int(np.median(counts))}（真实 {actual['trade_count']}）")
        print()

    if args.dump:
        pd.DataFrame(records).to_csv(args.dump, index=False)
        print(f"每次置换的结果已写入 {args.dump}")


if __name__ == "__main__":
    main()

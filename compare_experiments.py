#!/usr/bin/env python3
"""Compare trained model bundles side by side.

Two numbers decide whether a change helped, and they can disagree: AUC says
whether the model predicts its label better, the backtest says whether that
prediction is worth money. A label change in particular can lower AUC and raise
profit at the same time - the old label was easier to predict precisely because
predicting it did not require knowing the direction.

The volatility share of feature importance is the third number to watch: it is
what exposed the original defect, where 88% of the model's weight sat on ATR
and range.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Features that describe how much a price moves, not which way.
VOLATILITY_FEATURES = {
    "atr_14_pct", "range_pct", "vol_12", "vol_24", "vol_ratio_12_24",
    "mkt_vol_12", "vol_ratio_to_mkt", "volx_ret_6", "volx_z_60",
}

# Features that describe this stock against something else.
CROSS_ASSET_PREFIXES = ("mkt_", "excess_", "beta_", "corr_", "rel_strength", "sector_", "volx_")


def load(path):
    import joblib

    return joblib.load(path)


def importance_share(bundle, predicate) -> float:
    rows = bundle.get("feature_importances") or []
    total = sum(row["importance"] for row in rows)
    if total <= 0:
        return 0.0
    selected = sum(row["importance"] for row in rows if predicate(row["feature"]))
    return selected / total


def gate_verdict(bundle) -> str:
    """Repeat the deployment gate in execution.py without needing a broker."""
    issues = []
    test_auc = (bundle.get("test_metrics") or {}).get("auc")
    if test_auc is None or test_auc < 0.60:
        issues.append(f"auc={test_auc}")

    walk_forward = bundle.get("walk_forward_summary") or {}
    folds = walk_forward.get("fold_count", 0)
    profitable = walk_forward.get("profitable_folds", 0)
    if folds < 2 or profitable < 2:
        issues.append(f"folds={profitable}/{folds}")

    sharpe = walk_forward.get("mean_sharpe")
    if sharpe is None or sharpe <= 0.50:
        issues.append(f"wf_sharpe={sharpe:.2f}" if sharpe is not None else "wf_sharpe=None")

    drawdown = walk_forward.get("worst_max_drawdown")
    if drawdown is None or drawdown < -0.10:
        issues.append(f"dd={drawdown}")

    return "PASS" if not issues else "REJECT (" + ", ".join(issues) + ")"


def describe(path, bundle) -> dict:
    model_config = bundle.get("model_config") or {}
    test_metrics = bundle.get("test_metrics") or {}
    test_backtest = bundle.get("test_backtest") or {}
    validation_backtest = bundle.get("validation_backtest") or {}
    walk_forward = bundle.get("walk_forward_summary") or {}

    name = Path(path).stem
    for noise in ("gradient_boosting_model", "exp_"):
        name = name.replace(noise, "")
    name = name.strip("_.") or Path(path).stem

    return {
        "name": name[:24],
        "label_mode": model_config.get("label_mode", "absolute"),
        "references": bundle.get("reference_symbols") or {},
        "features": len(bundle.get("feature_columns") or []),
        "cost_bps": model_config.get("transaction_cost_bps"),
        "entry": (bundle.get("thresholds") or {}).get("entry_probability"),
        "test_auc": test_metrics.get("auc"),
        "positive_rate": test_metrics.get("positive_rate"),
        "vol_share": importance_share(bundle, lambda f: f in VOLATILITY_FEATURES),
        "cross_share": importance_share(bundle, lambda f: f.startswith(CROSS_ASSET_PREFIXES)),
        "val_return": validation_backtest.get("total_return"),
        "val_sharpe": validation_backtest.get("sharpe"),
        "test_return": test_backtest.get("total_return"),
        "test_sharpe": test_backtest.get("sharpe"),
        "wf_folds": f"{walk_forward.get('profitable_folds', 0)}/{walk_forward.get('fold_count', 0)}",
        "wf_sharpe": walk_forward.get("mean_sharpe"),
        "wf_return": walk_forward.get("mean_total_return"),
        "gate": gate_verdict(bundle),
    }


def fmt(value, spec="", dash="-"):
    if value is None:
        return dash
    try:
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def cost_sweep(bundle, costs, split="test"):
    """Re-score one bundle's held-out predictions at several cost levels.

    A strategy's edge has a size, and that size is what decides whether a cost
    assumption is survivable. Reporting one number at one cost hides it; the
    break-even point is the honest summary.
    """
    from ibkr_ml.backtest import simulate_probability_strategy
    from ibkr_ml.config import RiskConfig

    rows = bundle.get(f"{split}_predictions")
    if rows is None or len(rows) == 0:
        return []

    thresholds = bundle.get("thresholds") or {}
    model_config = bundle.get("model_config") or {}
    stored_risk = bundle.get("risk_config")
    if stored_risk:
        from dataclasses import fields, replace

        known = {f.name for f in fields(RiskConfig)} - {"log_dir"}
        risk_config = replace(RiskConfig(), **{k: v for k, v in stored_risk.items() if k in known})
    else:
        risk_config = RiskConfig()

    results = []
    for cost in costs:
        result = simulate_probability_strategy(
            prediction_rows=rows,
            entry_probability=float(thresholds.get("entry_probability", 0.58)),
            exit_probability=float(thresholds.get("exit_probability", 0.48)),
            transaction_cost_bps=float(cost),
            max_active_positions=model_config.get("max_active_positions", 2),
            risk_config=risk_config,
        )
        results.append((cost, result))
    return results


def break_even_cost(sweep) -> float | None:
    """Linearly interpolate where total return crosses zero."""
    for (low_cost, low), (high_cost, high) in zip(sweep, sweep[1:]):
        if low["total_return"] >= 0.0 > high["total_return"]:
            span = low["total_return"] - high["total_return"]
            if span <= 0:
                return low_cost
            return low_cost + (high_cost - low_cost) * (low["total_return"] / span)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare trained model bundles side by side.")
    parser.add_argument("bundles", nargs="+", help="Paths to .joblib model bundles.")
    parser.add_argument(
        "--cost-sweep",
        nargs="*",
        type=float,
        default=None,
        help="Re-run each held-out backtest at these one-way costs in bps, e.g. 0 1 3 5 8.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="test",
        help="Which saved prediction set the cost sweep replays.",
    )
    args = parser.parse_args()

    rows = []
    for path in args.bundles:
        if not Path(path).exists():
            print(f"warning: {path} does not exist, skipped")
            continue
        rows.append(describe(path, load(path)))

    if not rows:
        print("No bundles to compare.")
        return

    print(f"{'experiment':<24} {'label':<18} {'feat':>5} {'cost':>6} {'AUC':>7} "
          f"{'pos%':>6} {'vol%':>6} {'xasset%':>8}")
    print("-" * 84)
    for row in rows:
        print(f"{row['name']:<24} {row['label_mode']:<18} {row['features']:>5} "
              f"{fmt(row['cost_bps'], '.0f'):>6} {fmt(row['test_auc'], '.4f'):>7} "
              f"{fmt(row['positive_rate'], '.1%'):>6} "
              f"{row['vol_share']:>6.1%} {row['cross_share']:>8.1%}")

    print()
    print(f"{'experiment':<24} {'val ret':>9} {'val Sh':>8} {'test ret':>9} {'test Sh':>8} "
          f"{'wf folds':>9} {'wf Sh':>8}")
    print("-" * 84)
    for row in rows:
        print(f"{row['name']:<24} {fmt(row['val_return'], '+.2%'):>9} {fmt(row['val_sharpe'], '.2f'):>8} "
              f"{fmt(row['test_return'], '+.2%'):>9} {fmt(row['test_sharpe'], '.2f'):>8} "
              f"{row['wf_folds']:>9} {fmt(row['wf_sharpe'], '.2f'):>8}")
    print("  (these columns use each bundle's own cost assumption; compare them")
    print("   only across bundles trained at the same cost, or use --cost-sweep)")

    print()
    print("Deployment gate:")
    for row in rows:
        print(f"  {row['name']:<24} {row['gate']}")

    if args.cost_sweep is not None:
        costs = args.cost_sweep or [0.0, 1.0, 2.0, 3.0, 5.0, 8.0]
        print()
        print(f"Cost sensitivity on the {args.split} split (one-way bps):")
        header = "  " + f"{'experiment':<24}" + "".join(f"{c:>9.0f}" for c in costs) + f"{'break-even':>12}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for path in args.bundles:
            if not Path(path).exists():
                continue
            bundle = load(path)
            sweep = cost_sweep(bundle, costs, args.split)
            if not sweep:
                print(f"  {Path(path).stem[:24]:<24} (no saved {args.split} predictions)")
                continue
            cells = "".join(f"{result['total_return']:>8.2%} " for _, result in sweep)
            crossing = break_even_cost(sweep)
            crossing_text = f"{crossing:.1f} bps" if crossing is not None else "-"
            print(f"  {describe(path, bundle)['name']:<24}{cells}{crossing_text:>11}")

    print()
    print("Legend: vol% = share of feature importance on volatility features")
    print("        xasset% = share on cross-asset features")
    print("        A label change can lower AUC and raise profit: the old label was")
    print("        easier to predict because predicting it needed no direction.")


if __name__ == "__main__":
    main()

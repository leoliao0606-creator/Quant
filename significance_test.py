#!/usr/bin/env python3
"""Ask whether a hold-out result could plausibly be luck.

A positive number on one period is not evidence by itself: with 172 trades a
coin flip can produce a respectable Sharpe. This reads the per-trade table
that holdout_validation.py wrote and runs two checks that need no further
access to the hold-out, so neither one spends the single permitted run.

bootstrap  - resample the trades with replacement many times and report how
             often the mean net return is below zero. That fraction is the
             probability of seeing this result if the trades were drawn from
             a population whose average is what we observed; a wide interval
             that straddles zero means the sample cannot tell us the sign.
sign flip  - randomly negate each trade's return, which is the null of "the
             model picks entries with no directional skill", and report the
             share of shuffles whose mean beats the real one. That share is
             an empirical p-value.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap and permutation test on hold-out trades.")
    parser.add_argument("trades_csv")
    parser.add_argument("--draws", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    import numpy as np
    import pandas as pd

    trades = pd.read_csv(args.trades_csv)
    if trades.empty:
        raise SystemExit("交易表为空，无法检验")
    # The equity curve is driven by dollars, not by an unweighted average of
    # percentage returns: conviction sizing puts more capital behind some
    # trades than others, so the two can even differ in sign. Test the dollar
    # series, which is what the reported total return actually is.
    returns = trades["pnl"].to_numpy(dtype=float)
    n = len(returns)
    observed = returns.mean()
    rng = np.random.default_rng(args.seed)

    # Bootstrap: how uncertain is the mean given only n trades?
    idx = rng.integers(0, n, size=(args.draws, n))
    boot_means = returns[idx].mean(axis=1)
    low, high = np.percentile(boot_means, [2.5, 97.5])
    share_negative = float((boot_means <= 0).mean())

    # Sign flip: the null of no directional skill.
    signs = rng.choice((-1.0, 1.0), size=(args.draws, n))
    null_means = (returns * signs).mean(axis=1)
    p_value = float((null_means >= observed).mean())

    per_symbol = trades.groupby("symbol")["pnl"].agg(["count", "mean"])
    winners = int((per_symbol["mean"] > 0).sum())

    print(f"交易数                {n}")
    print(f"每笔平均盈亏          {observed:+,.2f} 美元")
    print(f"未加权每笔收益率      {trades['net_return'].mean():+.4%}")
    print(f"每笔盈亏标准差        {returns.std(ddof=1):,.2f} 美元")
    print(f"盈亏合计              {returns.sum():+,.0f} 美元")
    print()
    print("自助法(bootstrap，有放回重抽样估计不确定性)")
    print(f"  95% 区间            [{low:+,.2f}, {high:+,.2f}] 美元/笔")
    print(f"  区间包含 0          {'是 —— 样本量不足以确定正负' if low <= 0 <= high else '否'}")
    print(f"  重抽样均值 <= 0 占比 {share_negative:.2%}")
    print()
    print("符号翻转检验(随机给每笔收益取正负，代表模型没有方向判断力)")
    print(f"  经验 p 值           {p_value:.4f}")
    print(f"  结论                {'显著优于无技能(p<0.05)' if p_value < 0.05 else '无法拒绝“纯属运气”'}")
    print()
    print(f"盈利标的数            {winners}/{len(per_symbol)}"
          f"   (集中在少数标的说明结果脆弱)")
    top = per_symbol.sort_values("mean", ascending=False)
    print(f"  最赚的三个(美元/笔) {[(s, f'{m:+,.0f}', int(c)) for s, (c, m) in top.head(3).iterrows()]}")
    print(f"  最亏的三个(美元/笔) {[(s, f'{m:+,.0f}', int(c)) for s, (c, m) in top.tail(3).iterrows()]}")

    contribution = trades.groupby("symbol")["pnl"].sum().sort_values(ascending=False)
    total = contribution.sum()
    if total != 0:
        top_share = contribution.head(3).sum() / total
        print(f"  前三名贡献了总盈亏的 {top_share:.1%}")


if __name__ == "__main__":
    main()

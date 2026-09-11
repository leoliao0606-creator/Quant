# Experiment protocol

Written before the experiments it governs, and not to be edited to match
results. Its whole purpose is to stop a search over many configurations from
being reported as a discovery: run enough variants and some will look good by
chance, so the bar has to be set while the outcome is still unknown.

## Goal

A strategy that clears every criterion below on **walk-forward validation**,
then survives one single evaluation on a final hold-out period that no
experiment has touched.

## Success criteria

All of them, not a majority. Measured on walk-forward folds, at 5 bps one-way
transaction cost, with the full live risk rules applied.

| Criterion | Threshold | Why this number |
|---|---|---|
| Profitable folds | **every fold** | One good fold out of three is noise |
| Mean Sharpe | **>= 1.0** | Below this is worse than a decent active fund |
| Mean annualised return | **> SGOV** (~4-5%) | SGOV is a 0-3 month Treasury ETF: the return available for taking no risk at all. Below it, the strategy is worse than doing nothing |
| Worst fold drawdown | **> -15%** | Deeper than this is not survivable in practice |
| Mean exposure | **>= 10%** | Guards against high Sharpe produced by an idle account: 0.7% exposure returning 1.9% a year is not a strategy |
| Trades per fold | **>= 100** | Statistical floor; fewer cannot separate skill from luck |
| Test AUC | **>= 0.52** | Weak but real ranking ability |

## Stopping conditions

**Success**: a configuration clears every criterion on walk-forward, and is
then confirmed once on the untouched final hold-out. That confirmation runs
exactly once per candidate. A candidate that fails it is dead - not retuned.

**Failure**: the pre-declared experiment matrix is exhausted with nothing
passing. That result gets reported as it is. Continuing to tune past the matrix
is how a search turns into curve fitting.

## Guarding against the multiple-comparison trap

1. **The final hold-out is quarantined.** The most recent 15% of the data is
   not read during any experiment. Model selection happens on
   train/validation; walk-forward measures stability; the hold-out is opened
   once, at the end.
2. **Every experiment is logged, including failures.** Twenty experiments where
   one passes at p<0.05 is one expected false positive, not a finding. The
   count has to be visible to interpret the winner.
3. **The bar rises with the number of attempts.** With N experiments run, a
   winner needs to clear the Sharpe criterion by a margin that scales with N -
   the same logic as a Bonferroni correction, applied by eye rather than
   pretending at precision.
4. **A result that only appears under one specific parameter is treated as
   noise.** A real effect degrades gradually as parameters move away from the
   optimum; a fitted one vanishes.

## What is not evidence

- A good validation-set number without walk-forward confirmation
- A high Sharpe with exposure below 10%
- A single profitable fold
- A result that requires the exact threshold found by the search, and breaks
  one grid step away
- Anything measured with the transaction cost below 5 bps one-way

## Experiment record

Every run appends to `experiments/results.jsonl`: configuration, all metrics,
pass/fail against each criterion, and a timestamp. Failures are recorded with
the same detail as successes.

## Hold-out result, run once on 2026-09-10

Model `artifacts/exp_Q_gpu.joblib`, unchanged. Period 2023-02-07 to
2025-04-02: 540 trading days, 2,888,805 rows, 69 symbols, never read by any
earlier experiment.

| | pre-registered | measured | |
|---|---|---|---|
| annualised return | > 5% | **+1.07%** | fail |
| Sharpe | >= 1.0 | **0.31** per-bar, 0.30 daily | fail |
| max drawdown | > -15% | -3.98% | pass |
| trades | >= 100 | 1257 | pass |

AUC 0.6257. Total return +2.29% over 2.14 years. Mean gross exposure 1.06%.
Sign-flip test p = 0.1877, bootstrap 95% interval on dollars per trade
[-2.11, +5.81] - the sample cannot establish that the return is not zero.
Against the stated goal of beating SGOV at roughly 4-5%, this is a loss.

The tuning period's walk-forward reported +7.40% annualised and Sharpe 1.69
over 54-day folds. The gap between that and +1.07% over 540 days is what
fitting to a short window buys.

### Why, in one number

Hold-out deciles by predicted probability show the ranking is real: the
bottom tenth returns +0.0038% over the next 12 bars and the top tenth
+0.0532%, correlation +0.656 across deciles, on rows the model never saw.

The edge is simply smaller than the cost of taking it:

| slice | gross return | net of 10 bps round trip |
|---|---|---|
| top 10% | +0.0532% | -0.0468% |
| top 1% | +0.0470% | -0.0530% |
| top 0.5% | +0.0913% | -0.0087% |
| top 0.1% | +0.2859% | +0.1859% |

Break-even round-trip cost is about 5 bps. Only the top 0.1% of bars - 2,889
out of 2.89 million - clears it, which is exactly why exposure settled at
1.06%. Lowering the entry threshold to trade more would lose money faster,
not slower.

The flat 5 bps per side was checked against IBKR's fixed schedule
($0.005/share, $1 minimum). 98.2% of these trades hit the $1 minimum and the
median position was only $879, so median commission is 11.4 bps per side; but
those trades are small in dollars, and the dollar-weighted total comes to
+3.08% with no spread and +1.75% at a 2 bps half-spread, against +2.29% at
the flat 5 bps. The cost assumption is fair and the conclusion does not turn
on it.

### The hold-out is now spent

Every number above came from one run of one configuration. Any parameter
chosen in light of it needs data this project has not read: either a period
earlier than 2023-02, or forward paper trading.

## What the audit found, and the split it forces (registered 2026-09-10)

The hold-out above was scored with the label the model was trained on:
"return over the next 12 bars exceeds +0.2%". That label mixes two questions,
and the model answered only one of them. Scoring the same predictions against
each question separately:

| question the AUC asks | AUC |
|---|---|
| does the return exceed +0.2% (the training label) | 0.6257 |
| is the return positive at all (direction) | **0.5024** |
| is the absolute move above its median (volatility) | 0.7145 |

Across probability deciles the up-rate moves from 51.29% to 52.21% and is not
monotonic, while the mean absolute move grows 6.34x, from 14.96 bp to 94.77
bp. The rising decile return in the table above is therefore a ~1pp
directional tilt amplified by a much larger move size, not a directional
signal. `build_target` in `ibkr_ml/features.py` warns about exactly this;
the warning turned out to be a description of the result.

Retraining with `--label-mode direction` confirmed it: mean walk-forward AUC
0.5095 over three folds, mean Sharpe -1.11, one fold profitable out of three.
Six of the top ten feature importances became symbol one-hot columns and two
became time-of-day columns - the model, denied the volatility shortcut, fell
back to fitting per-symbol and per-hour constants.

A second finding constrains where the signal could ever be traded: 15.5% of
all rows have their 12-bar horizon cross the session close, but 66.1% of the
top 1% of signals do, because volatility peaks near the close. Those
overnight signals earn +0.03 bp against +13.81 bp for the intraday ones. The
existing close buffer already blocks them, which is why gross exposure sat at
1.06%: two thirds of the model's strongest signals are untradeable by
construction.

Checks that passed: no lookahead (the only negative shifts in features.py are
the label at lines 412-413), no train/hold-out overlap, no split artefacts
(all 11 bars moving more than 15% are 09:30 earnings gaps, and NVDA's
2023-05-25 close of 37.638 confirms the cache is split-adjusted), and the
per-trade P&L sums to the reported total return.

One weakness worth stating: the bootstrap in `significance_test.py` resamples
trades as if independent. Concurrent positions share market exposure, so the
true interval is wider than the one reported - which only makes p = 0.1877
less significant, never more.

### Conclusion

At 5-minute bars with a 1-hour horizon, this feature set has no usable
directional edge. The +1.07% annualised was not bad luck.

### The daily-bar split, fixed before any result is seen

Cost is fixed per round trip; predictable move size grows with holding time.
An earlier scan pointed the same way - 1-day bars held one week scored AUC
0.5555 against 0.5164 for 5-minute bars held one hour - on only 1,475 rows,
which is too few to believe but enough to justify testing properly.

Twenty years of daily bars for all 70 symbols are now cached (5,025 rows per
symbol from 2006-09-18; V starts 2008-03-19 at its IPO). The periods are
assigned now, before anything is run:

| period | dates | use |
|---|---|---|
| tuning | 2006-09-18 .. 2016-12-31 | every experiment, walk-forward inside it |
| hold-out A | 2017-01-01 .. 2019-12-31 | scored once, for the one chosen configuration |
| hold-out B | 2020-01-01 .. 2022-12-31 | reserve, only if A is passed and the design changes |
| 2023 onward | 2023-01-01 .. today | not a hold-out: the 5-minute work already read this era |

`train_model.py --end-date 2017-01-01` enforces the tuning boundary.

Two benchmarks, not one. The universe was chosen in 2026 for liquidity, so
running it back to 2006 selects companies that survived and grew: a long-only
strategy on this basket inherits an upward bias that has nothing to do with
the model. Beating SGOV is therefore not sufficient here. The registered bar
is both of:

- annualised return above SGOV's ~4-5%, and
- Sharpe above equal-weight buy-and-hold of the same 70 symbols over the same
  period, measured at the strategy's own average gross exposure.

A strategy that clears the first but not the second is a worse way to hold
stocks, not an edge.

### An engine bug this exposed

`simulate_probability_strategy` flattened every position on the last bar of
each calendar day, matching the live loop. With daily bars every bar is the
last bar of its day, and the flatten branch in `generate_trade_decision` sits
ahead of every entry rule: the backtest would have returned zero trades, not
short holdings. `flatten_at_session_close` now defaults to reading the median
bar interval and only flattens intraday bars; three tests in
`tests/test_backtest.py::TestSessionCloseFlattening` hold the behaviour in
place.

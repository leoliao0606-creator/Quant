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

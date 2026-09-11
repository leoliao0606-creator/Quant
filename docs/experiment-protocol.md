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

## The daily-bar scan, inside the tuning window (2026-09-10)

Twenty years of daily bars for 70 symbols cost 2.5 minutes to fetch - a
single "20 Y" request returns fifteen years in 0.7 seconds, against 24
seconds per 90-day chunk of five-minute bars. Four symbols drop out of the
2006-2016 window: XLC launched in 2018, the current VXX contract was issued
in 2018 after the original matured, and IBKR simply has no PEP before
2017-12 or AVGO before 2016-02. Requesting those explicitly with an earlier
end date returns empty, so the depth limit is IBKR's, not the request's.
That leaves 66 symbols and 159,583 rows.

Same features, same universe, same risk rules; only the label and the
horizon change.

| label | horizon | valid AUC | test AUC | valid ann. | test ann. | valid expo | test expo |
|---|---|---|---|---|---|---|---|
| direction | 3 | - | - | +1.32% | +0.30% | 2.2% | 6.6% |
| direction | 10 | 0.5073 | 0.4991 | +1.72% | +0.98% | 9.8% | 7.1% |
| direction | 21 | 0.4969 | 0.5420 | +5.56% | +5.98% | 20.9% | 22.9% |
| market_relative | 10 | 0.5209 | 0.5191 | +9.03% | +7.55% | 32.7% | 35.8% |
| market_relative | 21 | 0.5157 | 0.5272 | +6.47% | +0.27% | 30.9% | 21.9% |

Longer horizons do help, which is what a fixed per-round-trip cost against a
growing predictable move predicts. But the 21-day direction run returned
+5.56% on a fold whose AUC was 0.4969 - worse than a coin - so the return
cannot be coming from the ranking, and that had to be resolved before any of
these numbers meant anything.

### The permutation control

`permutation_test.py` keeps every rule and replaces only the probabilities
with a permutation of themselves. Read the Sharpe column: permuting destroys
the persistence that keeps a slot filled, so a scrambled signal opens more
positions and runs at 33-73% exposure against the real 22%, and returns at
different exposures are not comparable.

| model | fold | real Sharpe | permuted median | p (Sharpe) | p (return) |
|---|---|---|---|---|---|
| direction 21 | validation | 1.54 | 0.90 | 0.0398 | 0.3881 |
| direction 21 | test | 1.92 | 0.47 | 0.0050 | 0.0597 |
| market_relative 10 | validation | 1.52 | 0.79 | 0.0348 | 0.1244 |
| market_relative 10 | test | 0.67 | 0.45 | 0.2786 | 0.4080 |

The 21-day direction model beats its own permutations on Sharpe in both
folds and on return in neither. Its contribution is not picking bigger
winners; it is reaching a similar return on a third to a half of the
exposure. Per unit of average exposure the test fold returns 0.357 against
0.085 for the permuted median.

Two confounds checked and dismissed. Turnover: the permuted runs make 324
trades against the real 264, and rerunning at `--transaction-cost-bps 0`
leaves the result unchanged (real Sharpe 1.98, permuted median 0.53, p =
0.0050), so commission explains none of the gap. Diversification: the
permuted portfolios hold *more* positions, and volatility grows with the
square root of the position count while exposure grows linearly, so they
should score a *higher* Sharpe for the same per-position edge. They score
lower.

The market-relative label produced the only pair of folds with matching AUC
(0.5209 and 0.5191), which looked like the most believable signal of the
five, and then failed the permutation test on its test fold at p = 0.2786.
Consistent AUC did not become a risk-adjusted edge.

### What this does not yet establish

Six configurations were run and the best was reported. Correcting for that
by multiplying the p-values by six, the 21-day test fold survives at 0.03
and its validation fold does not, at 0.24.

The profit is also concentrated: on the test fold AMD alone is 29% of P&L,
the top three are 60% and the top five 85%, and six take-profit exits supply
53%. The validation fold is better spread - 29 of 33 symbols profitable, top
three 45%.

The reproduction windows used for the permutation runs were taken from the
first and last trade dates rather than the fold boundaries, so they are
narrower than the backtests they correspond to and the two sets of numbers
should not be quoted together. Within each permutation run the real and
permuted arms share a window, so the p-values stand.

Hold-out A (2017-2019) stays untouched until one configuration survives more
folds inside the tuning window.

### Registered before running: hold-out A

The 21-day direction configuration passed every check the tuning window can
supply. Six walk-forward folds instead of three: 6 of 6 profitable, mean AUC
0.5264, mean Sharpe 1.46. The stop-loss sweep redone with position size held
constant (see known-issues #11 - the stop is the denominator of the sizing
formula, so the first sweep varied size, not stops) keeps all eight fold
results positive, including the arm with no stop and no take-profit at all:
+3.66% and +2.51%, Sharpe 1.19 and 0.83. The result is not an artefact of
one risk setting.

Scored now, once, on 2017-01-01 to 2019-12-31.

The model is `artifacts/daily_h21.joblib` exactly as trained - no refit. It
learned from 2006-09-18 to roughly 2014-02 (the first 70% of the tuning
rows) and its thresholds were chosen on 2014-02 to 2015-05. Refitting on all
tuning data would mean choosing a new train split and a new way to set
thresholds, and those choices would be made having seen the tuning results.
Leaving three years of data unused makes the test harder, not easier.

Configuration: label direction, horizon 21 bars, minimum holding 21 bars,
quadratic conviction sizing, 10 concurrent positions, max_position_fraction
0.20, risk_per_trade 0.01, stop 8%, take-profit 15%, max daily loss 5%, 5 bps
per side, references SPY and XLK, 66 symbols.

It passes only if all of:

| criterion | required |
|---|---|
| annualised return | > 5% |
| Sharpe | >= 1.0 |
| max drawdown | > -15% |
| trades | >= 100 |
| Sharpe vs equal-weight buy-and-hold at the same average exposure | higher |
| permutation p on Sharpe, within-timestamp | < 0.05 |

The last two exist because the first four can be met without a model: the
2026 universe run backwards drifts upward on its own, and a fold with AUC
0.4969 still returned +5.56%.

Hold-out B (2020-2022) stays closed regardless of the outcome.

### Hold-out A, scored once on 2026-09-10

`artifacts/daily_h21.joblib`, unmodified, on 2017-03-30 to 2019-12-31.
44,418 rows, 673 trading days, 66 symbols.

| criterion | required | measured | |
|---|---|---|---|
| annualised return | > 5% | **+2.91%** | fail |
| Sharpe | >= 1.0 | **0.67** | fail |
| max drawdown | > -15% | -10.43% | pass |
| trades | >= 100 | 502 | pass |
| Sharpe vs buy-and-hold at 18.42% exposure | higher | **0.67 vs 1.19** | fail |

AUC 0.4925. Total return +7.97% over 2.75 years, average gross exposure
18.42%, peak 65.5%.

Equal-weight buy-and-hold of the same basket, scaled to the same 18.42%
average exposure, returned +3.08% at Sharpe 1.19 over the same days. The
strategy lost to doing nothing on both axes. Holding the basket and holding
SPY (+2.13%, Sharpe 0.86) both beat it risk-adjusted.

One limitation of the run: the frames were loaded starting 2017-01-01, so
the 60-bar feature warm-up consumed January to March and the scored period
begins 2017-03-30 rather than 2017-01-03. Loading the warm-up from 2016
would have recovered about 60 trading days. That is a second read of the
hold-out, so it was not done.

The tuning window said 6 of 6 folds profitable, mean Sharpe 1.46, and
permutation p between 0.005 and 0.04. The hold-out says AUC 0.4925 and a
Sharpe below buy-and-hold. This is the same shape as the five-minute
result: strong in tuning, absent out of sample. Two independent attempts -
different bar size, different horizon, different label, ten times the
history - reached the same place.

### Conclusion on the whole approach

Technical features computed from OHLCV bars, fed to a gradient-boosted tree,
thresholded into a long-only position, do not produce a directional edge on
US large caps at any horizon tested from one hour to one month. The tuning
periods keep producing Sharpe above 1; the hold-outs keep producing nothing.
The gap is fitting error, and it has now been measured twice.

What the models do predict is volatility: AUC 0.7145 against 0.5024 for
direction on the intraday data. That skill has a use that needs no options
and no directional call - sizing. Tested on the tuning window with the
crudest possible forecast, trailing realised volatility and no model at all:

| | annualised | volatility | Sharpe | max drawdown |
|---|---|---|---|---|
| equal-weight buy-and-hold | +12.39% | 21.66% | 0.65 | -49.08% |
| scaled by 20-day realised vol, cap 1x | +12.31% | 16.35% | 0.79 | -34.84% |
| scaled by 20-day realised vol, cap 2x | +17.15% | 22.34% | 0.82 | -39.21% |
| scaled by 60-day realised vol, cap 1x | +11.62% | 16.74% | 0.74 | -34.10% |

Same return, a quarter less volatility, fifteen points less drawdown, from a
forecast that took one line of pandas. The project has been discarding the
one thing its models are good at.

#### The permutation control on the hold-out

| arm | real | permuted median | p |
|---|---|---|---|
| within-timestamp, Sharpe | 0.67 | 0.82 | 0.7015 (140/200) |
| within-timestamp, return | +2.91% | +5.01% | 0.8607 (172/200) |
| global, Sharpe | 0.67 | 0.90 | 0.7711 (154/200) |
| global, return | +2.91% | +9.26% | 0.9801 (196/200) |

On the tuning window the model beat its own permutations on Sharpe at p =
0.005 to 0.04. On the hold-out a random reassignment of the same
probabilities does *better*: the permuted median return is +5.01% against
the model's +2.91%, and 140 of 200 random arrangements reached a higher
Sharpe. The signal is not weak out of sample, it is worse than none.

All six registered criteria fail. Hold-out A is spent; hold-out B (2020-2022)
stays closed.

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

### Volatility targeting, tested against its own control (2026-09-10)

The claim that the project should pivot to volatility needed the same
treatment as everything else, so it got a control: compare a
volatility-scaled book not against the unlevered basket but against a
*constant* position of the same average size. Scaling exposure changes
return and volatility together and leaves Sharpe alone, so any comparison
that lets average exposure move is measuring leverage, not timing.

All figures net of 5 bps per side on turnover.

| period | scaled Sharpe | constant-exposure control | difference | drawdown scaled vs control |
|---|---|---|---|---|
| 2006-2016, 20-day, cap 1x | 0.78 | 0.64 | **+0.139** | -35.0% vs -46.1% |
| 2006-2016, 20-day, cap 2x | 0.80 | 0.64 | **+0.151** | -39.6% vs -64.1% |
| 2006-2016, 60-day, cap 2x | 0.73 | 0.64 | +0.083 | -36.4% vs -61.8% |
| 2017-2019, 20-day, cap 1x | 1.39 | 1.36 | +0.024 | -13.1% vs -16.5% |
| 2017-2019, 20-day, cap 2x | 1.42 | 1.36 | +0.056 | -17.1% vs -23.3% |
| 2017-2019, 60-day, cap 2x | 1.28 | 1.36 | **-0.088** | -18.3% vs -20.2% |

The Sharpe gain is the 2008 crash. Without a crisis in the window it falls
to +0.02 to +0.06 and one variant is negative. The drawdown reduction
survives in both periods and is the honest benefit.

One idea tested and rejected: weighting each stock by the inverse of its own
volatility, which is where a per-stock forecast would help most. It *lowers*
Sharpe, 0.64 to 0.54, because it tilts toward low-volatility names and in a
basket selected in 2026 for size and liquidity those are the laggards.
Combining it with portfolio-level targeting is worse than targeting alone
(0.69 against 0.80). The per-stock volatility skill the models have does not
convert into a better portfolio this way.

### Where this leaves the goal

Without alpha, "beat SGOV" reduces to "take equity risk", and the margin is
proportional to the risk taken. Sizing a volatility-targeted basket to 5%
annual volatility returns about +3.5% over 2006-2016 and much more over
2017-2019 - the difference being the equity risk premium realised in each
window, which is the thing that cannot be forecast. Volatility targeting
improves the shape of that trade-off, mostly by cutting the worst drawdown,
and creates no return of its own.

Two directional attempts have now failed on untouched data. The remaining
honest options are: accept equity risk at a chosen size with volatility
targeting for drawdown control and no model at all; or find information that
is not in the price series, which is not obtainable from the IBKR bar feed
this project uses.

## Post-earnings drift: tested and rejected (2026-09-11)

Price and volume had failed twice at predicting direction, so the next
attempt changed the question rather than the model: after a company reports
and the market reacts strongly upward, does the stock keep outperforming?
Both inputs are recoverable from the bars - an earnings day is a large move
on heavy volume repeating quarterly, and the market's reaction on the day is
the surprise.

On the tuning window it looked like the best result this project had
produced. +20.21% annualised at Sharpe 0.89 against the same basket at the
same 72.9% exposure returning +9.38% at Sharpe 0.65; beta 0.964, so not a
leverage tilt; +10.33% annualised alpha at t = 2.18; a permutation keeping
the entry days and randomising the symbols beaten 200 times out of 200; 19
of 21 rolling two-year windows positive; a cluster bootstrap over symbols
keeping alpha positive in 500 of 500 draws.

It does not survive.

### What killed it

The detector was improved using an outside fact rather than a sweep: US
companies almost all report before the open or after the close, so a report
reaches the tape as an overnight gap, not as intraday range. Detectors were
then judged on how well they recover the real reporting calendar - the
third to sixth week after each quarter ends, which covers 34.6% of trading
days - and never on what they earned.

| detector | events/symbol-year | in reporting season | vs random |
|---|---|---|---|
| intraday move > 3.0x, volume > 1.8x | 2.65 | 46.0% | 1.33x (z=10.1) |
| overnight gap > 3.0x, volume > 1.8x | 2.60 | **51.8%** | **1.50x (z=15.2)** |
| overnight gap > 2.0x, volume > 1.8x | 2.89 | 48.6% | 1.40x (z=13.0) |

The gap detector finds earnings better at the same event count. And the
alpha runs the other way:

| signal | threshold | events/year | alpha | t |
|---|---|---|---|---|
| overnight gap | 3.0 / 1.8 | 2.60 | +2.16% | 0.53 |
| overnight gap | 2.0 / 1.5 | 3.48 | +0.10% | 0.02 |
| intraday move | 3.0 / 1.8 | 2.65 | **+10.33%** | **2.18** |
| intraday move | 2.0 / 1.5 | 3.73 | +4.26% | 0.84 |

The better the detector identifies actual earnings, the less alpha there
is. Post-earnings drift predicts the opposite, so whatever the +10.33% was,
it was not that.

Three facts together settle it. One cell of a 2x2 reaches t = 2.18 while
the other three sit at or below 0.84, and Bonferroni over just those four
puts it at p = 0.12 - before counting the move thresholds, volume
thresholds, rank floors and holding periods swept earlier. The
theoretically motivated detector removes it. And on 2017-2019, scored from
one continuous event history so the rank floor stays calibrated, alpha is
+4.65% at t = 0.76 with permutation p = 0.10 and 0.14.

A false positive produced by testing many configurations, found by asking
what the signal was supposed to be and then checking whether it was that.

## Five more approaches, and the one that survives (2026-09-11)

### Cross-sectional signals: reversed out of sample

The feature set had never reached past 24 bars, so the best-documented
equity anomaly - twelve-month momentum skipping the most recent month - had
never been tested. It was registered as the primary hypothesis and five
other cross-sectional signals were run alongside it, so that picking the
best afterwards would be visibly not what happened. Top 13 of 66 names,
rebalanced every 21 days, 5 bps per side, alpha measured against the
equal-weight basket.

| signal | tuning alpha | t | 2017-2019 alpha | t |
|---|---|---|---|---|
| momentum 12-1 (primary) | +4.67% | 1.53 | **-7.01%** | -1.32 |
| momentum 6-1 | +2.00% | 0.68 | -4.97% | -0.92 |
| momentum 12-0 | +5.97% | 1.92 | -4.07% | -0.76 |
| short-term reversal | -9.03% | -2.76 | +1.50% | +0.29 |
| low volatility | -2.95% | -1.46 | -1.94% | -0.69 |
| trend, price vs 200-day mean | +7.04% | 2.17 | +2.81% | 0.50 |

On the tuning window the pattern was coherent: every momentum-like signal
positive, every contrarian one negative, which reads as more than noise.
Out of sample all three momentum alphas turn negative and reversal turns
positive. The signs flip. Coherence within one window is not evidence of
persistence, and treating it as such was a mistake.

### Overlays on an index, where survivorship cannot reach

Every result above is computed on 66 symbols chosen in 2026 for size, so no
statistic inside that set separates an effect from the selection. SPY and
QQQ have no such problem, and neither overlay selects anything.

SPY, 2006-09 to 2026-09, cash credited at 4.3%:

| | annualised | volatility | Sharpe | max drawdown |
|---|---|---|---|---|
| buy and hold | +9.16% | 19.51% | 0.55 | -56.46% |
| 200-day trend filter | +6.62% | 11.37% | 0.62 | -18.85% |
| volatility targeting | **+9.39%** | 14.90% | **0.68** | -40.09% |
| both | +6.91% | 11.10% | 0.66 | -17.59% |

The trend filter halves the drawdown and costs two and a half points of
return over twenty years; it earns its keep only in 2020-2022 (Sharpe 0.77
against 0.34) and is expensive in every calm stretch. Volatility targeting
improves Sharpe in five of eight period-instrument pairs, ties one, loses
two, and raises the return slightly rather than cutting it.

### The overnight split: real, and untradeable

Almost all of the long-run equity return arrives between the close and the
next open. Over twenty years SPY's overnight leg returns +5.46% at Sharpe
0.51 and its intraday leg +3.52% at Sharpe 0.31; for QQQ the split is
+10.19% at 0.82 against +4.81% at 0.35.

Holding only overnight means two trades a day, 504 round trips a year. At 1
bp per side SPY's +5.46% becomes +0.27%; at 2 bps it becomes -4.66%. QQQ
survives 1 bp at +4.78% and dies at 2 bps. The effect is real and this
route to it is not.

It does explain something retrospectively: the five-minute work was hunting
direction inside the session, which over twenty years carries a Sharpe of
0.31 on SPY and almost no drift.

### Volatility targeting, held to the same standard

It is the only thing left, so it got the treatment that killed the others.

| | SPY | QQQ |
|---|---|---|
| buy and hold Sharpe | 0.547 | 0.762 |
| volatility targeted | 0.658 | 0.867 |
| constant position, same average exposure | 0.547 | 0.762 |
| **same weights in random order** | **0.460 (p = 0.0020)** | **0.673 (p = 0.0020)** |
| block bootstrap improvement | +0.072 [-0.024, +0.180] | +0.063 [-0.017, +0.158] |

The shuffled-weight control is the informative one. It keeps average
exposure, turnover and the whole distribution of position sizes, and breaks
only the pairing of a small position with a high volatility forecast.
Breaking it costs 0.20 of Sharpe, and none of 500 shuffles reached the real
number. The mechanism is real.

The block bootstrap - 21-day blocks, so the volatility clustering the
strategy feeds on is preserved rather than destroyed - puts the improvement
at +0.06 to +0.07 with about a tenth of resampled paths showing none. Real,
modest, and consistent with what the literature reports.

### Where the project stands

Six approaches tested against untouched data or a proper control: machine
learning on intraday bars, the same on daily bars, post-earnings drift,
cross-sectional momentum and its relatives, trend timing, and the overnight
split. None produces alpha that survives. One risk-shaping technique does
survive, and it forecasts nothing.

Without alpha, beating SGOV reduces to taking equity risk at a chosen size,
and volatility targeting improves the shape of that trade by about 0.1 of
Sharpe. Beating the S&P 500 needs either more risk or information this
project does not have. Any figure here that appears to beat the S&P 500 by
holding the 66-symbol basket should be read as the survivorship bias it is:
that basket returned +12.38% against SPY's +5.25% over 2006-2016 with no
strategy at all.

### A better volatility forecast does not make a better portfolio (2026-09-11)

Targeting runs on the crudest forecast there is, the trailing 20-day
standard deviation, so the obvious improvement is a better forecaster.
Five were compared on 2006-2016 - 20-day, 60-day, EWMA at RiskMetrics'
lambda, a 20/60 blend, and HAR-RV, which regresses the coming month's
volatility on the last day, week and month of it and is the benchmark most
published models fail to beat. HAR-RV was chosen: best Sharpe on SPY (0.53
against 0.47), within 0.05 on QQQ, and drawdowns of -27.9% and -26.1%
against roughly -41% for everything else, because it forecasts the next
month rather than measuring the last one and so de-risks before the move.

On 2017-2026 the choice did not hold:

| SPY 2017-2026 | annualised | Sharpe | max drawdown |
|---|---|---|---|
| buy and hold | +13.38% | 0.78 | -34.10% |
| 20-day realised, cap 1.0 | +11.80% | **0.86** | **-21.70%** |
| HAR-RV, cap 1.5 (chosen) | +13.74% | 0.84 | -24.37% |

| QQQ 2017-2026 | annualised | Sharpe | max drawdown |
|---|---|---|---|
| buy and hold | +20.23% | 0.92 | -35.62% |
| 20-day realised, cap 1.0 | +18.26% | **0.99** | **-30.80%** |
| HAR-RV, cap 1.5 (chosen) | +19.17% | 0.93 | -33.96% |

HAR-RV is the better volatility forecaster and the worse position sizer.
For deciding how much to hold, the crude window already carries what
matters.

The same table is the first positive result in this project to hold across
two separate periods with the parameter fixed in the first: 20-day
targeting improves Sharpe and cuts the worst drawdown on both instruments,
over ten years that include the 2020 crash and the 2022 bear market.

### What beating the index actually comes to (2026-09-11)

Targeting raises Sharpe and lowers both return and volatility, so the
comparison has to be made at one level of risk: lever the targeted book
until its volatility matches the index, and charge the borrowing. The
leverage factor is fixed on 2006-2016 and applied unchanged afterwards.

Idle cash earns 4.3% and borrowed cash costs 6.0%, IBKR's retail margin
rate. Charging the bill rate both ways flatters every levered variant by
the spread times the borrowed fraction, which at 1.33x is about half the
measured advantage.

Unlevered, all four instrument-period pairs move the same way:

| | Sharpe | max drawdown | annualised |
|---|---|---|---|
| SPY 2006-2016 | 0.35 → **0.49** | -56.5% → **-40.9%** | +5.26% → **+6.60%** |
| SPY 2017-2026 | 0.78 → **0.88** | -34.1% → **-21.2%** | +13.38% → +12.15% |
| QQQ 2006-2016 | 0.60 → **0.72** | -53.5% → **-39.4%** | +11.11% → **+11.58%** |
| QQQ 2017-2026 | 0.92 → **1.01** | -35.6% → **-30.0%** | +20.23% → +18.68% |

Levered to matched volatility with financing paid, the return advantage
mostly disappears: SPY 2017-2026 gives +13.81% against +13.38% at
identical Sharpe, QQQ +21.49% against +20.23%. The drawdown advantage
survives on SPY and does not on QQQ.

Nine equal-weighted sector ETFs, the only diversification available
without bond data, are worse than SPY in both periods before and after
targeting. Sector spreading adds nothing here.

One thing that came out of getting the leverage test wrong first: the 1.0
cap is doing more work than the targeting. Removing it so the rule can
borrow during calm stretches turns SPY's selection-period Sharpe from 0.47
into 0.27. The useful rule is "hold less when volatility is high", not
"size to a constant volatility".

### The answer to the original question

Beat SGOV: yes, with room. SPY volatility-targeted returned +12.15% over
2017-2026 at a worst drawdown of -21.2%, against SGOV's roughly 4.3%.

Beat the S&P 500: on risk yes, on return no. After realistic financing the
levered version is a wash - +0.43 points a year at identical Sharpe. The
real gain is the unlevered one: about 1.2 points a year of return given up
for a drawdown a third smaller.

That is not alpha. It is a better-shaped way to hold the same risk, it is
the only result here whose parameter was fixed in one period and confirmed
in another, and it is free of the survivorship problem that contaminates
every result computed on the 66-symbol basket.

### The same fixed rule on thirteen index funds (2026-09-11)

Every constant was already set - 16% target, 20-day window, cap 1.0, 5%
band - so the rule can be pointed at the rest of the cache without fitting
anything further. Index funds only: their membership follows rules rather
than hindsight, so the survivorship problem in the single-stock work does
not apply here.

| | Sharpe improved | Sharpe worse | drawdown improved |
|---|---|---|---|
| 2006-2016 | 13 of 13 | 0 | 13 of 13 |
| 2017-2026 | 10 of 13 | 3 | 13 of 13 |

The three that lose Sharpe are IWM (-0.05), XLE (-0.06) and XLB (-0.10),
all in the later period, and all three still cut their worst drawdown by
13 to 22 points.

The clearest single case is XLF through the financial crisis: buy-and-hold
returned -1.69% annualised at a worst drawdown of -83.7%, and the same
rule returned +3.73% at -42.7%. Volatility rose before the collapse
finished, and the rule was already smaller.

Twenty-six instrument-period pairs, drawdown improved in every one, mean
improvement about 18 points. That is the strongest evidence in this
project, and it is evidence for a risk overlay rather than for a forecast.

### Correction: the Sharpe gain was mostly an accounting artefact

The breadth table above reports Sharpe improving in 23 of 26 pairs. That
number is wrong and the reason is worth recording.

Sharpe was computed as mean over standard deviation of the portfolio's
total return, with idle cash credited at 4.3%. Cash has no volatility, so
adding interest to a series raises its mean without touching its
denominator: a book held entirely in cash scores infinity. Volatility
targeting holds 80 to 93% on average against buy-and-hold's 100%, so it
was collecting a bonus for the cash it happened to hold rather than for
anything it did.

Recomputed on return above cash, the picture changes:

| | Sharpe improved | median change | annual return | drawdown improved |
|---|---|---|---|---|
| 2006-2016, bills at 0.6% | 12 of 13 | +0.080 | -0.31% | 13 of 13 |
| 2017-2026, bills at 2.2% | 6 of 13 | -0.007 | -2.03% | 13 of 13 |

The bill rate matters because the targeted book's excess return is exactly
`weight x (buy-and-hold excess) - costs`: a higher assumed rate shrinks
what is being scaled, so giving up exposure costs more. Assuming today's
4.3% across a decade when bills paid nearly nothing understates the
strategy; assuming zero overstates it. Across 0% to 4.3% the count runs
from 20 of 26 down to 15 of 26 and the median gain from +0.053 to +0.008.

What survives every version of the calculation is the drawdown: improved
in 26 of 26 pairs and in all twenty parameter settings tested, by a median
of 17 points.

So the honest statement is narrower than the one above it. Volatility
targeting is a drawdown tool. Its risk-adjusted gain is concentrated in
the window containing a prolonged bear market and is absent from the one
without - the same shape as trend timing - but it costs two points a year
rather than six to nine, and it never goes fully flat.

### Mean reversion at the index level, and why it is not added

Cross-sectional short-term reversal is negative in this universe: buying
the stocks that fell most over a month lost 9% a year against the basket.
Index-level reversion is a different claim and it is present. Grouping
days by the index's trailing three-day move and reading the excess over
the unconditional average:

| bottom quintile, excess in bp | 1 day | 3 days | 5 days | 10 days |
|---|---|---|---|---|
| SPY 2006-2016 | +11.0 | +27.6 | +43.7 | +52.9 |
| SPY 2017-2026 | +5.9 | +17.2 | +21.9 | +33.9 |
| QQQ 2006-2016 | +15.6 | +27.3 | +35.1 | +33.5 |
| QQQ 2017-2026 | +12.8 | +21.7 | +23.1 | +24.2 |

Present in all four, and present whether volatility was high or low at the
time, so it is not a restatement of the volatility effect.

This matters because it contradicts the shipped rule. A decline raises
measured volatility and the rule cuts exposure; these numbers say the days
after a sharp decline pay better than average. So a tilt was tested:
`weight = clip(target/volatility - beta x trailing 3-day return, 0, 1)`.

On 2006-2016 beta = 5 is a clear improvement - Sharpe better than
buy-and-hold in 13 of 13 against 12 of 13, median gain +0.144 against
+0.080, and annual return moving from -0.31% to +0.67%, so it beats
buy-and-hold on return as well as risk. On 2017-2026 the same beta gives 6
of 13 and a median of -0.024, slightly worse than no tilt at all. The
shipped rule keeps beta = 0.

### What the rule actually does, episode by episode

Four techniques have now shown the same shape, and "works in a crisis" is
too loose an explanation because the later window contains two crashes.
SPY, the same fixed rule, scored over dated episodes:

| episode | shape | buy and hold | rule | difference | drawdown held → rule |
|---|---|---|---|---|---|
| 2008 decline | slow | -56.1% | -36.1% | **+20.1%** | -56.5% → -36.5% |
| 2009 recovery | rebound | +62.4% | +31.3% | **-31.1%** | -8.0% → -7.7% |
| 2011 euro crisis | medium | -7.8% | -9.9% | -2.1% | -19.5% → -16.8% |
| 2015-16 | medium | -1.7% | -2.3% | -0.7% | -14.4% → -12.8% |
| 2018 Q4 | fast | +0.9% | -0.2% | -1.1% | -20.2% → -16.7% |
| 2020 covid | very fast | +3.7% | +1.6% | -2.1% | **-34.1% → -18.0%** |
| 2022 bear | slow | -14.5% | -12.7% | +1.7% | -25.4% → -21.0% |
| 2025 pullback | - | +8.9% | +5.1% | -3.7% | -19.0% → -15.3% |

The entire return advantage is 2008, and most of it is handed back in
2009. Taken together, which is the only fair way since one does not happen
without the other, buy-and-hold compounds to -28.7% across the two and the
rule to -16.1%.

The drawdown improves in all eight, including every episode where the rule
loses on return. 2020 is the clearest: 2.1 points of return given up, and
the worst drawdown cut from -34.1% to -18.0%.

So the accurate description is: it pays one to four points a year in calm
markets for a smaller drawdown in every kind of decline. Its return
advantage appears only in slow grinding bears, it loses ground in V-shaped
recoveries, and over a full cycle it comes out ahead.

### Two attempts to fix the recovery weakness, and why neither is adopted

The episode table says the rule's one real cost is the rebound: after
March 2009 realised volatility stayed high for months while the market
rallied 62%, and the rule sat at 30 to 40%. Total volatility cannot tell a
market falling violently from one rising violently. Two single changes,
each argued from design rather than fitted:

  downside volatility   the standard deviation of negative days only,
                        rescaled by root-two. A rally with large up-days
                        raises total volatility and leaves this alone.
  de-risk only below    apply the scaling when price is under its 100-day
  the trend             average and hold full size otherwise.

| variant | 2006-2016 Sharpe beats hold | 2017-2026 | 2017-2026 median | 2009 rebound |
|---|---|---|---|---|
| total volatility (shipped) | 12 of 13 | 6 of 13 | -0.007 | +30.8% |
| downside volatility | 12 of 13 | **10 of 13** | **+0.031** | +32.2% |
| only below the trend | 11 of 13 | 7 of 13 | +0.019 | **+39.1%** |

Downside volatility looks like a clear win on the verification window. The
selection window cannot separate the three, so preferring it now would be
choosing on the verification window - the mistake this project keeps
finding in its own earlier work.

A cleaner read was available. Which of two sizing rules is better is a
relative question, and survivorship inflates both identically, so the 55
individual stocks are a fresh sample for it:

| | downside better than total | median | sign test |
|---|---|---|---|
| 2006-2016, 53 stocks | 26 of 53 | -0.000 | p = 1.0000 |
| 2017-2026, 55 stocks | 35 of 55 | +0.021 | p = 0.0581 |

One period is a coin flip and the other is marginal; 61 of 108 overall.
The ETF result did not replicate. **The shipped rule keeps total
volatility.**

The same table says something worth keeping about where the rule belongs.
On individual stocks over 2017-2026 it improves Sharpe in only 20 of 55
and costs 4.23% a year at the median, against 6 of 13 and 2.03% on the
index funds - single stocks jump, so the rule cuts too much - while the
drawdown still improves in 55 of 55. It is an index tool.

## The survivorship bias, measured (2026-09-11)

Every single-stock result in this project was benchmarked against an
equal-weight basket of 66 symbols chosen in 2026 for size and liquidity.
198 more were downloaded, built the opposite way: companies that were
large in 2006 and have gone sideways or down since - GE, IBM, INTC, T, VZ,
F, XRX, Nokia, the 2008 banks, the declining retailers. That gives a
second basket to measure the first against.

| | symbols | annualised | Sharpe | max drawdown |
|---|---|---|---|---|
| 2006-2016, original 66 | 53 | +14.01% | 0.70 | -47.6% |
| 2006-2016, wide | 241 | +12.39% | 0.64 | -51.3% |
| 2017-2026, original 66 | 55 | **+19.27%** | 1.01 | -33.9% |
| 2017-2026, wide | 253 | **+12.35%** | 0.71 | -38.8% |
| 2017-2026, SPY | 1 | +13.38% | 0.78 | -34.1% |

The old benchmark was inflated by 1.6 points a year over 2006-2016 and
6.9 over 2017-2026. The recent figure is the larger one because the
companies a 2026 screen selects are exactly those that led the last
decade.

It is a lower bound. Seventeen of the 215 requests failed, and the reason
is the point: WBA taken private, GPS renamed, JWN taken private, X bought
by Nippon Steel, ANSS by Synopsys, JNPR by HPE, HES by Chevron, MRO by
ConocoPhillips, K renamed Kellanova. Companies that were large and then
stopped trading are unreachable from a live data feed, and they are
precisely the tail this exercise is trying to restore.

One more thing the table says: over 2017-2026 the equal-weight basket of
253 stocks returned less than SPY and at a lower Sharpe. Equal-weighting a
wide list of large companies underperformed simply owning the
capitalisation-weighted index, which is the mega-cap concentration of the
last decade seen from the other side. For the stated goal, SPY is a better
starting point than any equal-weight stock basket assembled here.

## Registered before running: twelve cross-sectional signals (2026-09-11)

Written before the run. The point of writing it first is that with twelve
signals and two periods there are twenty-four cells, and at the usual 5%
threshold more than one will look significant by chance alone.

### Why run this at all

Look-ahead bias runs both ways. It made momentum 12-0 look like +4.87% when
it earns +1.64%, and it made short reversal look like -6.80% when it earns
-0.90%. Any signal rejected here on a number produced by the broken
pipeline was rejected on the wrong number, and a signal the bias pushed
*down* would have been dismissed without a second look. So the whole family
is re-scored on the repaired pipeline, and the family is widened at the
same time, because the marginal cost of one more signal is now a line of
code.

### The twelve

Momentum family, already re-scored, carried along as a reference:

  1. momentum 12-1      `close[-21] / close[-252]`
  2. momentum 12-0      `close / close[-252]`
  3. momentum 6-0       `close / close[-126]`

New, each with a published basis:

  4. residual momentum      12-1 return less beta times the basket's, so a
                            name is not credited for moving with the market
  5. volatility-scaled mom  momentum 12-1 divided by 126-day volatility
  6. 52-week-high proximity `close / max(close, 252)`
  7. MAX effect             minus the largest single-day return of the last
                            21 days; lottery-like names are said to
                            underperform
  8. idiosyncratic vol      minus the 126-day standard deviation of the
                            residual from the basket
  9. dollar-volume trend    minus (21-day dollar volume / 252-day dollar
                            volume); names whose attention has just spiked
 10. return skew            minus the 126-day skew of daily returns
 11. low beta               minus the 252-day beta against the basket
 12. acceleration           3-month momentum less the prior 9-month

Every signal is evaluated at the close of D-1 and earns from day D onward,
holdings drift with price between rebalances, and trading happens only on a
rebalance date. Top 20% of the universe, 21-day rebalance, 5 bps a side.

### What counts as a pass

All four, not the best of four:

  a. Alpha against the equal-weight basket of the same universe is positive
     in **both** 2006-2016 and 2017-2026. A signal that only works in one
     half is the momentum mistake again.
  b. Newey-West t on full-period alpha exceeds **2.87**. That is 0.05/12
     two-sided: twelve signals are being tested, so the threshold moves.
  c. The Sharpe beats a random control that draws only from names trading
     that day, at p < 0.05.
  d. Alpha stays positive at 20 bps a side.

2017-2026 is no longer a clean hold-out; it was read during the momentum
work. That is why (a) requires both halves rather than using one to choose
and the other to confirm - the second read cannot carry the weight a first
read would have.

### What is expected

Nothing passes. Six signals have already been re-scored and the largest
surviving full-period t is below 1. This is being run to close the question
rather than in expectation of finding an anomaly, and a clean negative
across twelve published effects is itself the answer to whether stock
selection is where this project should spend its remaining time.

## Result: none of the twelve passed (2026-09-11)

Run with `python cross_section_audit.py --family --draws 100`. Criteria as
registered above; the letters mark which of (a) both halves positive,
(b) Newey-West t over 2.87, (c) random control p under 0.05, (d) alpha
still positive at 20 bps were met.

| signal | 2006-2016 | 2017-2026 | full | t(NW) | Sharpe | p | 20 bps | met |
|---|---|---|---|---|---|---|---|---|
| momentum 12-1 | -2.01% | +4.38% | +1.19% | +0.60 | 0.64 | 0.168 | +0.32% | ...d |
| momentum 12-0 | -1.44% | +4.83% | +1.69% | +0.84 | 0.66 | 0.059 | +0.86% | ...d |
| momentum 6-0 | +0.88% | +4.52% | +2.71% | +1.40 | 0.72 | 0.010 | +1.58% | a.cd |
| residual momentum | -2.77% | +4.17% | +0.61% | +0.33 | 0.63 | 0.248 | -0.28% | .... |
| vol-scaled momentum | -3.03% | +2.34% | -0.38% | -0.21 | 0.57 | 0.762 | -1.31% | .... |
| 52-week-high | -3.11% | +0.99% | -1.09% | -0.69 | 0.51 | 1.000 | -2.75% | .... |
| MAX effect | -2.81% | -2.80% | -2.77% | -2.13 | 0.44 | 1.000 | -4.97% | .... |
| idiosyncratic vol | -0.51% | -1.07% | -0.59% | -0.40 | 0.58 | 0.693 | -1.13% | .... |
| dollar-volume spike | -0.80% | -1.08% | -0.92% | -0.69 | 0.60 | 0.485 | -2.91% | .... |
| return skew | -0.94% | +1.08% | +0.04% | +0.03 | 0.65 | 0.109 | -0.94% | .... |
| low beta | -0.93% | -0.59% | -0.81% | -0.44 | 0.47 | 1.000 | -1.11% | .... |
| acceleration | -2.06% | -3.88% | -2.93% | -1.58 | 0.48 | 1.000 | -4.32% | .... |

The random control's full-period Sharpe runs 0.52 to 0.71 with a median of
0.60, and nine of the twelve signals land inside it.

The best of them, momentum 6-0, clears three of four and fails the one that
was set to account for having tried twelve: t = 1.40 against a threshold of
2.87. Without the correction it would read as p < 0.05 and this section
would say something different, which is the reason the threshold was
written down first.

One pattern is worth recording. All four momentum variants - 12-1, 12-0,
6-0 and residual - read about -2% over 2006-2016 and about +4.5% over
2017-2026. Four signals cannot break at the same date by coincidence; the
2008-2009 momentum crash sits inside the first window and nothing like it
sits inside the second. That makes the split a single event rather than a
stable effect, and it also means the +4.5% column is a description of a
decade without a crash, not a forecast of one.

**Stock selection is closed.** Twelve published cross-sectional effects,
253 US names, twenty years, one repaired pipeline, and a threshold set in
advance: nothing. Combined with the six direction models that failed their
hold-outs, the evidence that this project cannot predict which stocks will
outperform is now about as strong as this dataset can make it. What remains
predictable here is risk, not return.

## RETRACTED: the momentum result above was look-ahead bias (2026-09-11)

Everything in the section that follows is withdrawn. The alphas in it were
produced by a backtest that let the signal read a price the account could
not have seen, and the controls that appeared to confirm them could not
detect that, because they shared the same fault. Kept here rather than
deleted: how the error survived four separate checks is the more useful
record.

### The fault

The weights started earning on the day the signal was read. `xsec_wide.py`
computed the signal on day D, set that day's weights from it, and collected
day D's return. For a signal containing day D's closing price that is
look-ahead bias - choosing the names using a price that had not printed
when the trade was supposed to happen. The headline signal, momentum 12-0,
is `closes / closes.shift(252)`, which contains it.

### Size of the error

`cross_section_audit.py` scores every signal three ways: A as-run, B with
the signal shifted a day so it uses only what had printed by the previous
close, C additionally letting the holdings drift with prices between
rebalances instead of being reset to equal weight daily at no cost.

Alpha against the equal-weight basket of the same 253 names:

| signal | contains day D's close | 2017-2026 A / B / C | 2006-2016 A / B / C |
|---|---|---|---|
| momentum 12-1 | no | +2.95% / +1.71% / +1.33% | -0.95% / -1.57% / -1.93% |
| momentum 12-0 | yes | **+4.87%** / +1.94% / +1.64% | **+1.14%** / -1.38% / -1.72% |
| momentum 6-0 | yes | +6.47% / +2.81% / +2.58% | +4.01% / +1.26% / +0.92% |
| price / MA200 | yes | +5.50% / +2.37% / +2.18% | +1.96% / -0.49% / -0.86% |
| short reversal | yes | -6.80% / -0.90% / -0.93% | -9.45% / -4.71% / -5.43% |
| low volatility | mildly | -1.26% / -1.64% / -1.75% | -0.92% / -1.00% / -1.12% |

The four signals containing day D's close lose about three points a year
when the day is taken away. Momentum 12-1, which does not contain it, loses
1.24 points, well under half a standard error of its own alpha, which is
noise. Short reversal moves the other way and gains 5.9 points, which is
what look-ahead predicts for a signal that buys whatever fell today and
then collects today's fall. The direction is right for every row, so this
is the bias and not a coincidence.

Corrected, momentum 12-0 earns -1.72% a year over 2006-2016 and +1.64% over
2017-2026, t below 1 in both. **Cross-sectional momentum does not work on
this universe.**

### The permutation test certified the bug

The original run reported p = 0.0099 from 100 permutations. Two faults:

1. The control drew from every column, including names that had not listed
   yet, whose price is NaN. Those slots earned nothing, so the control was
   never fully invested and was too easy to beat. Drawing only from names
   trading that day lifts the control's median Sharpe to 0.65 with a range
   of [0.51, 0.78] - against the real signal's corrected 0.67. p = 0.36.
2. More seriously, the permutation replaced the *selection* while keeping
   the same timing machinery. The look-ahead lived in the machinery, so it
   was present in the real arm and absent from the control. The test was
   measuring the bias.

This is the general lesson and it applies to every control in this file: a
permutation, a bootstrap or a cost sweep built on one backtest pipeline
cannot see a fault in that pipeline. They all agreed because they were all
asking the same question of the same wrong object. What found it was
recomputing the result a different way, not testing the result harder.

### What survives

The volatility overlay does, and on new data. With momentum's selection
reduced to noise, the pair over 2006-2026 on 253 names reads:

| arm | annual | volatility | Sharpe | drawdown |
|---|---|---|---|---|
| equal-weight basket | +12.44% | 20.69% | 0.60 | -51.31% |
| momentum selection | +12.21% | 20.64% | 0.59 | -49.82% |
| momentum + volatility target | +10.26% | 14.86% | 0.63 | **-31.90%** |

The selection arm is worth nothing - it matches the basket it is drawn
from. The overlay still removes 18 points of drawdown, and over 2017-2026
alone it takes -34.08% to -17.35%. That reproduces on a 253-stock portfolio
the shape established on thirteen index funds: drawdown much better, Sharpe
better early and flat late, return lower. It is an independent confirmation
on a different kind of book, and it is the only thing in this project that
has now been confirmed twice.

Reproduce with `python cross_section_audit.py --start 2017-01-01` and
`python cross_section_audit.py --overlay`.

### Also withdrawn

The survivorship story below - the narrow universe's +5.97% against the
wide universe's +1.10% - used the same day-zero timing in `xsec.py`
(`usable.iloc[i]` selects and earns on the same day), so the +5.97% is
inflated by roughly the same three points. The measured gap between the
narrow and wide baskets, 1.61% and 6.92% a year, does not depend on any
signal and stands. The four-cell story built on top of it does not.

## Momentum, retested on a universe that contains losers (2026-09-11)

Cross-sectional momentum was rejected earlier on the 66-symbol universe:
positive over 2006-2016 and negative out of sample. With roughly 250
symbols the four cells read differently.

| universe | 2006-2016 alpha (t) | 2017-2026 alpha (t) |
|---|---|---|
| 66 symbols picked in 2026 | +5.97% (1.92) | -4.07% (-0.76) |
| ~250 symbols incl. laggards | +1.10% (0.39) | **+6.66% (1.76)** |

One mechanism fits all four: momentum works by avoiding losers, and a
universe assembled in 2026 from the winners contains none to avoid. The
earlier rejection was reading the absence of losers as the absence of an
effect. The weak 2006-2016 figure is the documented momentum crash - 2008
cost 10.7 points of alpha and 2009 another 7.4 - and the other eight years
of that window average about +2.7%.

The 2017-2026 result passes the battery that rejected the earnings work:

  costs        alpha +6.66% at 5 bps a side, +5.85% at 20 bps, turnover
               5.4x a year
  years        8 of 10 positive, the exceptions 2017 (-6.6%) and 2021
               (-13.0%)
  clustering   300 symbol bootstrap draws, alpha median +6.73%, 5-95%
               [+4.69%, +8.36%], none negative, and Sharpe below the
               basket in none

Against that: t = 1.76 is not significant on its own, the same universe
gives +1.10% in the earlier window, and the cell was found by looking at
four. Over twenty years the honest expectation is nearer +3% than +6.7%,
with crash risk attached.

### Momentum and the volatility overlay together

The two survivors fail in opposite conditions. Momentum's crash is a
violent rebound off a bottom, which is when the overlay is holding a third
of a position, so the overlay should remove momentum's worst years - a
prediction, not a parameter.

| 2006-2026, ~250 symbols | annualised | volatility | Sharpe | drawdown |
|---|---|---|---|---|
| equal-weight basket | +12.44% | 20.69% | 0.60 | -51.31% |
| momentum only | +14.78% | 20.77% | 0.70 | -48.10% |
| **momentum + overlay** | +12.39% | 14.90% | **0.76** | **-30.34%** |

| crash years | basket | momentum | combined |
|---|---|---|---|
| 2008 | -31.35% | -33.04% | **-18.91%** |
| 2009 | +46.95% | +17.45% | +14.24% |
| 2021 | +29.83% | +25.81% | +14.15% |

Half right. The overlay removed 14 points of the 2008 loss; it could not
help in 2009, where momentum captured +17.45% of the basket's +46.95%
because the selection was wrong, not the sizing.

Read against its own universe - the only fair comparison, since both arms
share it - the combination delivers the same return at Sharpe 0.76 against
0.60 and a drawdown of -30% against -51%.

The absolute figure is not trustworthy. The 253-symbol basket returned
+12.44% over twenty years against SPY's +9.16%, and that gap is the
survivorship bias. The relative improvement survives it because both arms
hold the same names; the level does not. Trading this forward needs a
universe defined by a rule - index membership - rather than by a list
written in 2026.

## Multi-asset diversification, finally testable

Bond, gold and commodity funds were downloaded today; every earlier
portfolio result was equities against cash because the cache held nothing
else. Monthly rebalancing, Sharpe on return above cash.

| 2006-2026 | annualised | volatility | Sharpe | drawdown |
|---|---|---|---|---|
| SPY | +9.16% | 19.51% | 0.47 | -56.46% |
| SPY + volatility overlay | +8.58% | 13.66% | 0.56 | -36.81% |
| **SPY + AGG + GLD, equal** | +7.02% | 9.60% | **0.60** | **-24.30%** |
| SPY + AGG, equal | +4.93% | 10.13% | 0.38 | -32.15% |
| seven assets, equal | +5.17% | 14.93% | 0.31 | -44.33% |

Stocks, bonds and gold in thirds is the best simple mix in both
sub-periods. Inverse-volatility weighting is worse everywhere because it
loads on bonds, which paid little across this particular twenty years.

It does not translate into beating the index. Levering the mix to SPY's
volatility needs 2.03x, and at IBKR's 6% margin rate that is
2 x 7.02% - 6% = 8.04%, below SPY's 9.16%. Better risk-adjusted return
that financing costs undo.

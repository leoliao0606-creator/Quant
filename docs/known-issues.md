# Known issues in the current approach

Written after the walk-forward passed 6 of 7 criteria, before acting on that
result. A list of what is still wrong is worth more at that moment than at any
other, because a passing number is exactly when the temptation to stop looking
is strongest.

Ordered by how much each could invalidate the result.

## 1. The result has no clean hold-out (severity: high)

Every parameter - stop at 4%, position 10%, gross cap 100%, hold 12 bars,
entry at the top 1% - was chosen by looking at the same 360 days the result is
measured on. The test split has been read dozens of times.

Walk-forward mitigates this but does not remove it: each fold retrains and
reselects its threshold, yet all three folds live inside the window whose
parameters were tuned.

**Status**: fetching 720 days so the earlier year can serve as a hold-out that
no experiment has touched. Until that runs, +7.40% annualised is a hypothesis.

## 2. Fills are priced at the decision bar's close (severity: high)

The backtest buys at the close of the bar that triggered the signal. That
close is only known once the bar has ended, so a live market order fills in
the next bar at an unknown price. At a 0.1% edge per trade, a few basis points
of slippage per fill is a large fraction of the result.

**Fix**: add `next_open` to the feature frame and fill there. Not done.

## 3. No purge between train and validation (severity: medium)

The label reads 12 bars ahead, so the last 12 rows of the training split have
labels built from prices that appear in validation. With 69 symbols that is
828 contaminated rows out of ~200k - small, but it inflates validation, and
validation is what selects the threshold.

**Fix**: drop `horizon_bars` rows either side of every split boundary.

## 4. Splits are cut by row count, not by timestamp (severity: medium)

Rows are concatenated across symbols and sorted by time, then split by
position. A boundary can fall inside one timestamp, putting some symbols of
that bar in train and the rest in validation.

## 5. Survivorship bias in the symbol list (severity: medium)

The 70 symbols are ones that are liquid and listed *today*. Anything that was
delisted, acquired, or collapsed during the period is absent, so the universe
is selected on having survived. This biases every backtest upward by an amount
nobody can measure from inside the data.

## 6. Transaction cost is a flat 5 bps (severity: medium)

A single number stands in for spread, commission and impact across SPY and
PLTR alike. Sensitivity analysis shows the result survives 2.5-8 bps, which
helps, but a per-symbol cost model would be closer to the truth - and the
strategy trades the more volatile names more often, where costs are highest.

## 7. Exposure floor in the protocol is arguably wrong (severity: low, but unresolved)

The criterion "mean exposure >= 10%" was written to catch high Sharpe produced
by an idle account. The strategy fails it at 0.81% while returning 11.79%
annualised with peak gross exposure at 98%: flat most of the time, nearly
fully invested when it acts. That is a timing strategy, not an idle one.

Recorded as a failure rather than rewritten after the fact. Whether the
criterion should become "peak gross exposure <= 100%" is a decision to take
deliberately, not one to slip in because it would turn a FAIL into a PASS.

## 8. Six months is a short window for judging decay (severity: low)

Signal edge by month ran +1.30, +0.30, +0.61, +0.26, -0.05, +0.18 percent.
First half averages +0.74%, second half +0.13%. That could be decay, or the
volatility regime (SPY per-bar volatility fell 43% over the same period), or
noise. Six points cannot separate them.

## 9. The model is one family with default-ish hyperparameters

Gradient boosting only, barely tuned. Not a defect in the result, but it means
"this is what the data supports" has not really been established - only "this
is what this model found".

## Resolved, kept here as a record

- Backtest ran different rules from the live loop → both now call
  `strategy.generate_trade_decision`
- Rules added to the backtest did not reach the live loop → gross cap, minimum
  holding and entry cutoff near the close are now in `execution.py`, with tests
- 5x gross leverage from `max_position_fraction * max_active_positions` →
  `max_gross_exposure` caps the book
- Thresholds that barely traded scored well by staying flat → activity floor
- Entry cutoffs searched as absolute probabilities → searched as percentiles
- Time-of-day features read the raw TWS clock → converted to US Eastern
- Orders held inside TWS were logged as submitted → acknowledgement check

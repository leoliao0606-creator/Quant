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

# IBKR ML Paper Trading Baseline

A trainable supervised-learning baseline for IBKR paper trading, replacing the
original news + LLM script.

**This is an intraday strategy, not a long-horizon one.** The model predicts
the return three 5-minute bars ahead - fifteen minutes - and the loop closes
every position before the session ends. Nothing is held overnight by default.

## Model design

- Market: liquid US equities and ETFs on IBKR paper account
- Data: historical OHLCV bars fetched from IBKR via `ib-insync`
- Bar size: `5 mins` by default, one decision cycle every 300 seconds
- Prediction target: set by `--label-mode`. `absolute` asks whether the return
  `horizon_bars` ahead clears a fixed threshold (3 bars and 0.1% by default);
  `volatility_scaled` scales that threshold with current ATR%; `direction` only
  asks which way it went. See [Choosing a label](#choosing-a-label) - this
  choice turned out to matter more than any feature.
- Model: `GradientBoostingClassifier`
- Validation: train/validation/test time split with automatic threshold selection
- Robustness check: walk-forward folds on later time windows
- Execution style: long/flat, probability-threshold entries, rule-based exits
- Session handling: entries stop at `--flatten-time`, open positions are closed
  before the close, and the book goes home flat
- Deployment guard: refuse unattended trading when model quality gates fail
  unless overridden

## Feature set

- Short-horizon returns: 1, 3, 6, 12 bars
- Gap and longer momentum: open-to-prev-close gap and 24-bar return
- Volatility: rolling standard deviation over 12 and 24 bars
- Volatility regime: short-vs-long volatility ratio and ATR%
- Trend: distance to 8- and 21-bar EMAs
- Trend structure: EMA spread and 20-bar breakout position
- Momentum: RSI(14)
- Intrabar shape: range percent and close location inside the bar
- Volume surprise: 20-bar z-score
- Time-of-day seasonality: sine/cosine encoding

Optional cross-asset features (`--market-symbol`, `--sector-symbol`,
`--volatility-symbol`) measure the stock against something else:

- Market return over 1, 6 and 24 bars
- **Excess return** - the stock's move with the market's move removed
- Beta and correlation to the market over 60 bars
- Relative strength: the stock/market ratio versus 20 bars ago
- Relative volatility, and the market's distance to its own 21-bar EMA
- Sector excess return and relative strength, and a volatility-proxy z-score

These matter because the single-asset block can only describe a stock's own
history. Given a label like "will it move more than 0.1%", the most predictable
thing in that block is volatility - and volatility says nothing about
direction. Relative performance is where directional information lives.

Reference series are aligned with `merge_asof(direction="backward")`, which
pairs each bar with the most recent reference bar *at or before* it. A
forward-filling join would pair a bar with a reference bar that had not
happened yet.

Bar timestamps arrive from TWS without a timezone, in the wall clock of the
machine TWS runs on. Features convert them to US Eastern first, so the
time-of-day encoding means the same thing wherever the data was fetched - but
only if `--bar-timezone` is declared on **both** `train_model.py` and
`paper_trade.py`. See [Timezones](#timezones).

## Risk controls

- Entry only when predicted probability exceeds `entry_probability`
- Exit when probability drops below `exit_probability`
- Portfolio cap via `max_active_positions`, with only the strongest signals allowed in
- **Bracket orders**: every entry is a market parent with a stop-loss and a
  take-profit child that live at IBKR, so the position stays protected if this
  process dies, loses its connection, or misses a cycle. Every close path
  cancels those children first, so nothing sells twice. Disable with
  `--no-bracket-orders`.
- In-process stop loss and take profit based on average entry cost, as a second
  layer on top of the bracket
- **Flatten before the close**: from `--flatten-time` (15:45 ET by default)
  until the session end the loop only closes positions. The flatten path does
  not consult the model or the data feed, so a stale bar cannot leave a
  position open overnight.
- Max position size capped by both notional exposure and risk budget. Note that
  `--risk-per-trade` only binds when it is below
  `max_position_fraction * stop_loss_pct` (0.20 * 0.008 = 0.0016 by default);
  above that every position is simply `max_position_fraction` of equity.
- Daily drawdown circuit breaker based on session start equity
- Daily trade cap, stale-data guard, duplicate-order guard, and structured
  JSONL logs for unattended operation
- **Order acknowledgement check**: after placing an order the loop waits for
  IBKR to acknowledge it. An order still in `PendingSubmit` was held inside TWS
  and never reached IBKR, so it is logged as `order_not_transmitted`, does not
  count against the daily trade cap, and prints the TWS setting to fix. Without
  this the log claimed trades that never happened.

## The backtest runs the live rules

`ibkr_ml/backtest.py` replays saved predictions through
`strategy.generate_trade_decision` - the same function the live loop calls. Stop
loss, take profit, the daily loss limit, the daily trade cap, position sizing
and the end-of-session flatten therefore all apply in the backtest. A rule
cannot exist on one side only.

This matters because `execution.py` reads Sharpe and drawdown out of that
simulation to decide whether a model may trade unattended. Two gaps remain, and
both make the result optimistic:

- Fills happen at the bar close the decision was made on. Live, that close is
  only known once the bar has ended, so a market order fills in the next bar.
- The backtest flattens on the last bar of each day rather than at
  `--flatten-time`, because bar timestamps carry no timezone.

Transaction cost defaults to **5 bps one-way** (`--transaction-cost-bps`),
covering the spread a market order crosses, IBKR commission and impact. The old
1 bps default understated it badly: the label is a 0.1% move, so a round trip
at 5 bps each way consumes the entire edge the model is trained to find. Keep
this honest - threshold selection reads it, and understating it picks entry
thresholds that only look profitable.

## Project layout

- `train_model.py`: fetches historical data from IBKR and trains the model
- `backtest_model.py`: replays the saved validation/test predictions without talking to IBKR
- `paper_trade.py`: loads the trained model and runs one cycle or a loop against the IBKR paper account
- `analyze_logs.py`: summarizes JSONL run logs and names why a run did not trade
- `compare_experiments.py`: compares trained bundles side by side, including a
  transaction-cost sweep and the break-even cost
- `ibkr_ml/`: feature engineering, training, signal generation, and IBKR execution
- `ibkr_ml/cache.py`: on-disk cache of fetched bars, so experiments are fast and comparable
- `tests/`: unit tests for the decision rules, the backtest, execution and the log analyzer
- `docs/raspberry-pi.md`: Raspberry Pi 4B setup and deployment notes
- `scripts/bootstrap_raspberry_pi.sh`: creates a Pi-friendly virtualenv and installs dependencies

## Quick start

1. Activate the environment. On this machine that is the `ibkr_trade` conda
environment (Python 3.10):

```bash
conda activate ibkr_trade
python -m pip install -r requirements.txt
```

Every `python` command below assumes it is active. If you are setting this up
somewhere else, a plain virtualenv works too:

```bash
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

On a Raspberry Pi, `scripts/bootstrap_raspberry_pi.sh` creates `.venv-pi`
instead; use that path there.

2. Run the tests. They need no IBKR connection and take under a second:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

3. Start IBKR TWS or IB Gateway in paper mode and enable API access.

**TWS must be told to let API orders through.** By default TWS checks every
API order against its order precautions (price percentage, total value limit,
size limit, number of ticks, and more) and holds anything that trips one behind
a dialog box, waiting for a human click. The order sits in `PendingSubmit`, no
error is returned, and it is discarded when the client disconnects - so an
unattended run places no trades at all and nothing says why.

In TWS: `File`/`Edit` -> `Global Configuration` -> `API` -> `Precautions` ->
tick **Bypass Order Precautions for API Orders**. TWS also offers this as a
one-off dialog the first time an API order trips a precaution; answering `Yes`
sets the same flag.

Also confirm `API` -> `Settings` -> **Read-Only API** is unticked, and use the
same `--client-id` across restarts so resting bracket children stay visible to
this process.

4. Train the first model:

```bash
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA
```

The risk arguments (`--stop-loss-pct`, `--take-profit-pct`,
`--max-daily-trade-count`, `--max-daily-loss-pct`, `--max-position-fraction`)
default to the same values `paper_trade.py` uses. Because the backtests inside
training replay those rules, **change them in both places or not at all** - a
model whose thresholds were selected against a 0.8% stop behaves differently
under a 2% one. The settings used are stored in the bundle, and
`backtest_model.py` replays with them.

For longer intraday histories, the loader fetches IBKR data in smaller chunks
automatically. The default 90 days leaves each walk-forward fold with about ten
trading days of out-of-sample data, which is too short to trust a Sharpe ratio
from; prefer a longer history:

```bash
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA \
  --duration "360 D" --max-duration-per-request "30 D"
```

If the API handshake is flaky, retry with a higher timeout and let the script
try new client IDs:

```bash
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA \
  --request-timeout 180 --connect-retries 5 --client-id 15 --client-id-step 1
```

Training also:

- Applies class balancing during model fit
- Selects entry/exit thresholds on the validation split
- Runs the live decision rules over the validation and test splits with costs
- Runs walk-forward folds to check stability across later periods
- Stores validation and test predictions, the risk settings, the bar timezone
  and the library versions in the model bundle

5. Review the held-out backtest:

```bash
python backtest_model.py --split test
```

It prints the risk rules it replayed, so the numbers can be read next to the
settings that produced them.

6. Run one dry paper-trading cycle:

```bash
python paper_trade.py --once --dry-run
```

7. Run one live paper-trading cycle that can send paper orders:

```bash
python paper_trade.py --once
```

8. Run continuous paper trading:

The continuous mode is the same script without `--once`. By default it runs one
cycle every `300` seconds and writes JSONL logs to `logs/`.

First, keep it in dry-run mode until data freshness, connectivity, and model
loading are stable:

```bash
python paper_trade.py \
  --symbols SPY QQQ AAPL MSFT NVDA \
  --host 127.0.0.1 \
  --port 7497 \
  --client-id 15 \
  --model-path artifacts/gradient_boosting_model.joblib \
  --risk-per-trade 0.01 \
  --max-position-fraction 0.20 \
  --max-active-positions 2 \
  --max-daily-trade-count 12 \
  --stop-loss-pct 0.008 \
  --take-profit-pct 0.015 \
  --max-daily-loss-pct 0.02 \
  --flatten-time 15:45 \
  --interval-seconds 300 \
  --log-dir logs \
  --dry-run
```

When that path is stable, remove `--dry-run` to allow paper orders.

Keep the **same `--client-id` across restarts**. Resting bracket children are
only visible to the client that placed them, so a restart under a different id
cannot cancel them before closing a position - which would sell the same shares
twice.

To keep it running after the terminal disconnects, wrap it with `tmux`:

```bash
tmux new -s quant
conda activate ibkr_trade   # or . .venv-pi/bin/activate on a Raspberry Pi
python paper_trade.py --symbols SPY QQQ AAPL MSFT NVDA --log-dir logs
```

Detach with `Ctrl-b` then `d`. Press `Ctrl-C` once to stop the loop; it exits at
the end of the current cycle instead of waiting out the remaining sleep interval.

## The bar cache

Pulling 360 days of five-minute bars for five symbols takes about ten minutes
and holds the TWS session throughout. Every experiment that only changes a
label or a feature would pay that again, and would work on a slightly different
window than the last run - so two results could not be compared.

Bars are therefore cached under `--cache-dir` (default `data_cache/`):

```bash
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA --duration "360 D" \
  --max-duration-per-request "30 D"          # first run downloads and caches
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA --duration "360 D" \
  --label-mode volatility_scaled              # second run reads the cache
```

A fully cached run **never opens a TWS session**, so it cannot collide with one
held on a phone or another machine. `--refresh-cache` re-downloads. The cache
prints its own age and warns when it is more than a few days old, because a
cache ends where it was taken, not today.

Timestamps are stored as UTC. A year of US bars spans a daylight-saving change,
so IBKR stamps part of the range `-04:00` and the rest `-05:00`; written out
with their local offsets, they come back as a mix pandas cannot parse into one
dtype.

## Choosing a label

The default `absolute` label asks whether the forward return beat a fixed
threshold. That question has a shortcut: in a volatile stretch **both**
directions clear a fixed bar more often, so a model can score well by
predicting *when* the market moves rather than *which way*. Trained on 360 days
of data that is exactly what happened - roughly three quarters of the feature
importance landed on ATR, range and rolling volatility, the model reached an
AUC of 0.67, and the strategy still lost money once costs were realistic.

The alternatives remove that shortcut:

- `volatility_scaled` compares the forward return with
  `--volatility-threshold-multiple` times current ATR%, so a volatile stretch
  needs a proportionally larger move to count as a positive.
- `direction` drops the size question and asks only for the sign.

Compare them directly, on identical cached data:

```bash
for mode in absolute volatility_scaled direction; do
  python train_model.py --symbols SPY QQQ AAPL MSFT NVDA --duration "360 D" \
    --label-mode "$mode" --market-symbol SPY \
    --model-path "artifacts/exp_$mode.joblib"
done

python compare_experiments.py artifacts/exp_*.joblib --cost-sweep 0 1 2 3 5 8
```

`compare_experiments.py` reports AUC next to the volatility share of feature
importance and the break-even transaction cost. Watch for AUC and profit moving
in opposite directions: the absolute label is *easier* to predict precisely
because predicting it does not require knowing the direction.

## Analyzing a run

After any session, ask the logs what happened:

```bash
python analyze_logs.py                 # every log under logs/
python analyze_logs.py --last 1        # the most recent one
python analyze_logs.py logs/paper_trade_2026-06-05.jsonl
```

It prints orders, decision reasons, bar-age statistics and errors, then names a
cause: `[DELAYED FEED]`, `[TIMEZONE]`, `[STALE]`, `[SESSION]`, `[TIMEOUT]`,
`[BLOCKED]`, `[NOT TRANSMITTED]`, `[TRADED]`. Use it before reading raw JSONL - "41 cycles ran and no
order was ever submitted, dominant reason stale_data" is the kind of thing that
is invisible when scrolling the file by hand.

The tool imports nothing beyond the standard library, so it also runs on a Pi
with no virtualenv activated.

### Reading the freshness log

Whenever a symbol is skipped for stale data, a future timestamp, or an
unchanged bar, a `bar_blocked` event is written with the timestamps and the
computed age:

```bash
grep bar_blocked logs/paper_trade_*.jsonl | tail -5
```

Compare `latest_bar_et`, `now_et`, and `age_minutes` to tell a delayed market
data subscription apart from a timezone mistake or a `--stale-after-minutes` set
too low. `analyze_logs.py` does that comparison for you.

Skipping a symbol does not skip its risk controls: stop loss, take profit, and
the daily loss limit are still evaluated for open positions, and can still send
a closing order. Only model-driven entries and exits are suppressed, because
the model probability is untrustworthy for that bar. When the price used for
that check came from a stale bar, the log entry carries `"price_is_stale": true`.

The end-of-session flatten ignores the freshness guard entirely. Closing the
book is not negotiable, so a stale bar must not be able to carry a position
overnight.

## Timezones

TWS returns bar timestamps without timezone information, in the wall-clock time
of the machine it runs on. Two things read that clock: the staleness guard and
the time-of-day feature.

Declare the zone on **both** commands whenever TWS is not on US Eastern:

```bash
python train_model.py --symbols SPY QQQ AAPL --bar-timezone UTC
python paper_trade.py --host 192.168.1.20 --bar-timezone UTC --once --dry-run
```

Without it the timestamps are read as US Eastern, so a Raspberry Pi or server on
UTC makes every bar look four to five hours off: the staleness guard rejects
everything, and the time-of-day feature lands in the wrong part of the session.
A model trained with one declaration and scored with another produces features
it never saw in training, silently. `paper_trade.py` prints a note when the
bundle's zone and the runtime zone differ.

A bar that lands in the future by more than `--future-bar-tolerance-minutes`
(default `1.0`) is reported as `future_bar_timestamp` rather than accepted as
fresh.

## Model bundles and library versions

A bundle is a joblib pickle of scikit-learn objects, so it only loads under a
compatible scikit-learn. The bundle in `artifacts/` loads under scikit-learn
1.7.x - the version in `ibkr_trade` - and fails under 1.9 with a bare
`ModuleNotFoundError: No module named '_loss'`. `requirements.txt` pins exact
versions for that reason, every bundle records the versions it was trained
under, and `load_model_bundle` reports drift or, when a load fails outright,
says which versions this machine has instead of the raw pickle error.

Retrain rather than chase versions:

```bash
python train_model.py --symbols SPY QQQ AAPL MSFT NVDA --duration "360 D" --max-duration-per-request "30 D"
python backtest_model.py --split test
python paper_trade.py --once --dry-run
```

The Pi bootstrap script deliberately prefers the system ARM packages over these
pins, because building the numeric stack from source there takes hours. That
makes a model trained elsewhere a version-drift candidate; the recorded
versions are what make it diagnosable.

## Raspberry Pi 4B

The codebase itself is portable to a Pi 4B. The practical differences are ARM
package installation and the lighter CPU budget during training.

- Fastest path: train on a stronger machine, copy
  `artifacts/gradient_boosting_model.joblib` to the Pi, and run only
  `paper_trade.py` there. Watch for the library-drift warning on first load.
- If the Pi should connect to TWS or IB Gateway on another machine, pass
  `--host <gateway-lan-ip>` and `--bar-timezone <that machine's zone>`.
- For a Pi-first setup guide, see [docs/raspberry-pi.md](docs/raspberry-pi.md).
- For a bootstrap script that creates `.venv-pi` and prefers system ARM
  packages when present, run `./scripts/bootstrap_raspberry_pi.sh`.

## Notes

- This is a baseline, not a production strategy.
- Do not connect it to a live account before adding more realistic slippage
  modeling, walk-forward retraining, and monitoring.
- The backtest now applies the live risk rules, but still fills at the decision
  bar's close. Treat it as a sanity check, not final evidence.
- The strategy re-enters immediately after a stop loss if the probability is
  still above the entry threshold. That is faithful to the live loop, and it
  also means the stop does not impose a cooling-off period.
- Training and validation splits are cut by row count without purging the
  `horizon_bars` rows whose labels straddle the boundary. The leakage is small
  but real.

# IBKR ML Paper Trading Baseline

This project replaces the original news + LLM script with a trainable supervised-learning baseline for IBKR paper trading.

## Model design

- Market: liquid US equities and ETFs on IBKR paper account
- Data: historical OHLCV bars fetched from IBKR via `ib-insync`
- Bar size: `5 mins` by default
- Prediction target: whether the next `horizon_bars` return is above a positive threshold
- Model: `GradientBoostingClassifier`
- Validation: train/validation/test time split with automatic threshold selection
- Robustness check: walk-forward folds on later time windows
- Execution style: long/flat, probability-threshold entries, rule-based exits
- Deployment guard: refuse unattended trading when model quality gates fail unless overridden

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

This is a stronger baseline than headline-driven prompting because it gives you:

- A train/test split
- Stable numeric features
- A repeatable model artifact
- Explicit risk controls before order placement

## Risk controls

- Entry only when predicted probability exceeds `entry_probability`
- Exit when probability drops below `exit_probability`
- Portfolio cap via `max_active_positions`, with only the strongest signals allowed in
- Stop loss and take profit based on average entry cost
- Max position size capped by both notional exposure and risk budget
- Daily drawdown circuit breaker based on session start equity
- Daily trade cap, stale-data guard, and structured JSONL logs for unattended operation

## Project layout

- `train_model.py`: fetches historical data from IBKR and trains the model
- `backtest_model.py`: replays the saved validation/test predictions without talking to IBKR
- `paper_trade.py`: loads the trained model and runs one cycle or a loop against the IBKR paper account
- `ibkr_ml/`: feature engineering, training, signal generation, and IBKR execution
- `docs/raspberry-pi.md`: Raspberry Pi 4B setup and deployment notes
- `scripts/bootstrap_raspberry_pi.sh`: creates a Pi-friendly virtualenv and installs dependencies

## Quick start

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Start IBKR TWS or IB Gateway in paper mode and enable API access.

3. Train the first model:

```bash
python3 train_model.py --symbols SPY QQQ AAPL MSFT NVDA
```

For longer intraday histories, the loader now fetches IBKR data in smaller chunks automatically.
If TWS is still slow, you can lower each request size explicitly:

```bash
python3 train_model.py --symbols SPY QQQ AAPL MSFT NVDA --duration "360 D" --max-duration-per-request "30 D"
```

If API handshake is flaky, retry with a higher timeout and allow the script to try new client IDs:

```bash
python3 train_model.py --symbols SPY QQQ AAPL MSFT NVDA --request-timeout 180 --connect-retries 5 --client-id 15 --client-id-step 1
```

The training step now also:

- Applies class balancing during model fit
- Selects entry/exit thresholds on the validation split
- Runs a simple long/flat backtest with transaction costs
- Runs walk-forward folds to check stability across later periods
- Stores validation and test predictions in the model bundle

4. Review the held-out backtest:

```bash
python3 backtest_model.py --split test
```

To make the backtest closer to live execution, keep the same max position cap:

```bash
python3 backtest_model.py --split test --max-active-positions 2
```

5. Run one dry paper-trading cycle:

```bash
python3 paper_trade.py --once --dry-run
```

6. Run one live paper-trading cycle that can send paper orders:

```bash
python3 paper_trade.py --once
```

## Raspberry Pi 4B

The codebase itself is portable to a Pi 4B. The practical differences are ARM package installation and the lighter CPU budget during training.

- Fastest path: train on a stronger machine, copy `artifacts/gradient_boosting_model.joblib` to the Pi, and run only `paper_trade.py` there.
- If the Pi should connect to TWS or IB Gateway on another machine, pass `--host <gateway-lan-ip>`.
- For a Pi-first setup guide, see [docs/raspberry-pi.md](docs/raspberry-pi.md).
- For a bootstrap script that creates `.venv-pi` and prefers system ARM packages when present, run `./scripts/bootstrap_raspberry_pi.sh`.

## Notes

- This is a baseline, not a production strategy.
- Do not connect it to a live account before adding more realistic slippage modeling, walk-forward retraining, and monitoring.
- The built-in backtest is intentionally simple; treat it as a sanity check, not final evidence.

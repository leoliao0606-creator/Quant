# Running on Raspberry Pi 4B

This project is portable to Raspberry Pi because the codebase is pure Python. The main constraints on a Pi 4B are package installation, IBKR desktop software placement, and CPU budget during training.

## Recommended topology

- Best fit: train the model on a faster machine, copy `artifacts/gradient_boosting_model.joblib` to the Pi, and use the Pi only for `paper_trade.py`.
- If TWS or IB Gateway is not on the Pi, keep it on another machine in the same LAN and pass `--host <gateway-lan-ip>` when running `train_model.py` or `paper_trade.py`.
- Training on the Pi is possible, but start with fewer symbols and shorter history to keep runtime and memory use under control.

## Base OS advice

- Prefer 64-bit Raspberry Pi OS.
- Make sure `python3 -V` is at least Python 3.11.
- If you plan to log continuously, prefer SSD storage over a microSD card.

## Bootstrap the environment

1. If you want to avoid building `numpy`, `pandas`, and `scikit-learn` from source on ARM, install the Debian packages first:

```bash
sudo apt update
sudo apt install -y python3-venv python3-numpy python3-pandas python3-sklearn
```

2. Run the project bootstrap script:

```bash
./scripts/bootstrap_raspberry_pi.sh
```

The script creates `.venv-pi`, reuses the system numeric packages when available, installs the remaining Python dependencies, and runs a small smoke test.

## Copy a trained model to the Pi

If you train elsewhere, copy only the model bundle you want to deploy:

```bash
mkdir -p artifacts
cp /path/to/gradient_boosting_model.joblib artifacts/
```

Then verify it on the Pi:

```bash
. .venv-pi/bin/activate
python backtest_model.py --split test --model-path artifacts/gradient_boosting_model.joblib
python paper_trade.py --once --dry-run --model-path artifacts/gradient_boosting_model.joblib
```

## Pi-friendly first commands

Dry run against a local TWS or Gateway session:

```bash
. .venv-pi/bin/activate
python paper_trade.py --once --dry-run
```

Dry run against a Gateway session on another machine:

```bash
. .venv-pi/bin/activate
python paper_trade.py --once --dry-run --host 192.168.1.20
```

Smaller first training job on the Pi:

```bash
. .venv-pi/bin/activate
python train_model.py --symbols SPY QQQ AAPL --duration "30 D" --max-duration-per-request "10 D" --walk-forward-splits 1
```

## Operational notes

- Keep `--interval-seconds 300` unless you have a reason to fetch data more often.
- Use `--once --dry-run` until the Pi can fetch bars, score the model, and write logs cleanly.
- If you see stale-data skips, check the Pi clock, IBKR connectivity, and whether the Gateway host is reachable on the configured port.
- If unattended paper trading is the goal, wrap `paper_trade.py` with `systemd` after the dry-run path is stable.

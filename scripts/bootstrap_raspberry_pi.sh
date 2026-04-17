#!/usr/bin/env bash
set -eu

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv-pi}"
USE_SYSTEM_SITE_PACKAGES="${USE_SYSTEM_SITE_PACKAGES:-1}"

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|armv7l|armv8l)
    ;;
  *)
    echo "warning: detected architecture '$ARCH', not a typical Raspberry Pi target." >&2
    ;;
esac

if [ ! -d "$VENV_DIR" ]; then
  if [ "$USE_SYSTEM_SITE_PACKAGES" = "1" ]; then
    "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
  else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
  fi
fi

# shellcheck disable=SC1090
. "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel

if python - <<'PY'
import importlib.util
import sys

required = ("numpy", "pandas", "sklearn")
missing = [name for name in required if importlib.util.find_spec(name) is None]
sys.exit(0 if not missing else 1)
PY
then
  echo "Using system-provided numeric stack from Raspberry Pi OS."
  python -m pip install joblib ib-insync
else
  echo "System numeric stack not found; installing full Python requirements with pip."
  if ! python -m pip install -r "$PROJECT_ROOT/requirements.txt"; then
    cat <<'EOF' >&2
Full pip install failed.

On Raspberry Pi OS, install the heavy numeric packages from apt first:
  sudo apt update
  sudo apt install -y python3-venv python3-numpy python3-pandas python3-sklearn

Then rerun:
  ./scripts/bootstrap_raspberry_pi.sh
EOF
    exit 1
  fi
fi

python - <<'PY'
import joblib
import numpy
import pandas
import sklearn

print("Python dependencies import cleanly.")
PY

python "$PROJECT_ROOT/test.py"

cat <<EOF

Bootstrap finished.

Activate the Raspberry Pi environment with:
  . "$VENV_DIR/bin/activate"

If TWS or IB Gateway is running on another machine, point the paper trader at it:
  python paper_trade.py --once --dry-run --host <gateway-lan-ip>

For a Pi-friendly first training run, start smaller:
  python train_model.py --symbols SPY QQQ AAPL --duration "30 D" --max-duration-per-request "10 D" --walk-forward-splits 1
EOF

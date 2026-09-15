#!/usr/bin/env bash
# Weekly run of the allocation runner, late in Friday's session.
#
# Weekly, not monthly. Over 2006-2026 checking the volatility scaling every
# week scores Sharpe 0.72 and checking it monthly scores 0.62, against 0.63
# for never scaling at all - at monthly the whole overlay buys nothing but a
# smaller drawdown. The cliff sits between weekly and fortnightly. See the
# table in portfolio_trade.py.
#
# CAPITAL is the slice of the account this strategy may use. The account also
# holds positions this project did not put there (VOO, QQQM, AMZN, GOOG,
# PLTR, MSTR, COIN as of 2026-09-11), and sizing off net liquidation while
# ignoring them would build a book on margin; the runner refuses to start
# unless this number is stated. Raise it only after those are cleared.
#
# Inside the session, not after it. The orders are market orders, and a market
# order sent after the close is not executed - it is parked until the next
# open. The runner refuses to send buys until the sells are done, so an
# after-close run would leave the book half rebalanced for a week. 15:30 gives
# thirty minutes of trading to fill in, and the runner checks with IBKR that
# the market is actually open before it sends anything.
#
# State lives in logs/overlay_state.json, which records the book scale this
# run adopted and the date the weights were last restored. The runner needs
# both to reproduce what was measured: the allocation is restored monthly
# (21 sessions) and the scale only moves when it has shifted more than 3%,
# so a run with no memory of the previous one would rebalance every week
# instead. logs/ is gitignored, so the file does not travel with the repo -
# deleting it, or running this on a fresh clone, makes the next run rebalance
# immediately. That is safe, it just pays one extra round of commission.
#
# To install:
#   (crontab -l 2>/dev/null; \
#    echo '30 15 * * 5 /home/cliao/Projects/Quant/run_weekly.sh') | crontab -
# The machine's clock is on America/New_York, so 15:30 is half an hour before
# the close all year, daylight saving included.
set -u
cd /home/cliao/Projects/Quant || exit 1

PYTHON=/home/cliao/miniconda3/envs/ibkr_trade/bin/python
CAPITAL=300000
LOG_DIR=logs
mkdir -p "$LOG_DIR"
STAMP=$(date +%Y-%m-%d)

{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    "$PYTHON" portfolio_trade.py \
        --allocation thirds \
        --capital "$CAPITAL" \
        --client-id 41 \
        --execute
    echo "退出码 $?"
} >> "$LOG_DIR/weekly_${STAMP}.log" 2>&1

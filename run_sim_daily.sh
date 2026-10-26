#!/usr/bin/env bash
# Daily run of the broker-free ledger, after the close.
#
# portfolio_sim.py keeps the weekly allocation going without a broker: the
# account is on IBKR Lite, which has no API, so portfolio_trade.py cannot
# reach a paper account. It decides on Fridays only, but it runs every
# weekday so that each session's dividends and mark are booked on the day.
# A missed day is caught up in order on the next run, and a decision
# computed late is the same decision, because it reads data through the day
# before the fill and nothing after.
#
# 18:30, not 16:00. Yahoo's daily bar keeps moving for a few minutes while
# the closing auction settles, and the script ignores a bar until half an
# hour after the close anyway.
#
# After it, signal_ledger.py books the three rule-driven ledgers that trade
# the pasted ChatGPT reports (logs/sim/chatgpt-daily, -radar, -short), and
# signals_control.py scores those reports against random picks. Each step
# runs whatever the one before it did; a failure is logged with its exit
# code and does not stop the rest.
#
# State and ledgers live in logs/sim/ and logs/signals/. logs/ is gitignored,
# so back them up: deleting a ledger ends its record, and starting again
# starts a new one from the day it is recreated, never from an earlier date.
#
# To install (the machine's clock is on America/New_York):
#   (crontab -l 2>/dev/null; \
#    echo '30 18 * * 1-5 /bin/bash /home/cliao/Projects/Quant/run_sim_daily.sh') | crontab -
# Calling it through /bin/bash means it runs even if the execute bit is lost,
# which is exactly how run_weekly.sh sat in the crontab for weeks without
# running once.
set -u
cd /home/cliao/Projects/Quant || exit 1

PYTHON=/home/cliao/miniconda3/envs/ibkr_trade/bin/python
LOG_DIR=logs/sim
mkdir -p "$LOG_DIR"

{
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %Z') ====="
    "$PYTHON" portfolio_sim.py --allocation thirds
    echo "退出码 $?"
    for task in daily radar short; do
        "$PYTHON" signal_ledger.py --task "$task"
        echo "退出码 $?"
    done
    "$PYTHON" signals_control.py
    echo "退出码 $?"
} >> "$LOG_DIR/daily_$(date +%Y-%m).log" 2>&1

"""Print the commands for the usual workflow and report what is set up.

Not a test suite despite the name - the real tests are under tests/ and run
with pytest.
"""

from pathlib import Path


MODEL_PATH = Path("artifacts/gradient_boosting_model.joblib")
LOG_DIR = Path("logs")


def main() -> None:
    print("IBKR ML paper-trading baseline.")

    if MODEL_PATH.exists():
        print(f"Model bundle found: {MODEL_PATH}")
        print("If it fails to load, the scikit-learn version has moved on; retrain.")
    else:
        print("No trained model found yet.")
        print("Train one with: python3 train_model.py --symbols SPY QQQ AAPL")

    log_files = sorted(LOG_DIR.glob("paper_trade_*.jsonl")) if LOG_DIR.exists() else []
    print(f"Run logs in {LOG_DIR}/: {len(log_files)}")

    print()
    print("Run the unit tests (no IBKR connection needed):")
    print("  python3 -m pytest")
    print("Preview one paper-trading cycle without orders:")
    print("  python3 paper_trade.py --once --dry-run")
    print("Evaluate the saved model on held-out predictions:")
    print("  python3 backtest_model.py --split test")
    print("Send orders to the IBKR paper account:")
    print("  python3 paper_trade.py --once")
    print("Find out why a run did or did not trade:")
    print("  python3 analyze_logs.py --last 1")


if __name__ == "__main__":
    main()

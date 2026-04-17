from pathlib import Path


MODEL_PATH = Path("artifacts/gradient_boosting_model.joblib")


def main() -> None:
    print("IBKR ML paper-trading baseline is ready.")
    if MODEL_PATH.exists():
        print(f"Model bundle found: {MODEL_PATH}")
    else:
        print("No trained model found yet.")
        print("Train one with: python3 train_model.py --symbols SPY QQQ AAPL")

    print("Preview one paper-trading cycle without orders:")
    print("python3 paper_trade.py --once --dry-run")
    print("Evaluate the saved model on held-out predictions:")
    print("python3 backtest_model.py --split test")
    print("Send orders to the IBKR paper account:")
    print("python3 paper_trade.py --once")


if __name__ == "__main__":
    main()

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Mapping

from .backtest import select_probability_thresholds, simulate_probability_strategy
from .features import FEATURE_COLUMNS, build_labeled_rows, build_latest_feature_row


def _load_joblib():
    try:
        import joblib
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'joblib'. Install requirements.txt first.") from exc
    return joblib


def _load_pandas():
    try:
        import pandas as pd
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'pandas'. Install requirements.txt first.") from exc
    return pd


def _load_numpy():
    try:
        import numpy as np
    except ModuleNotFoundError as exc:
        raise RuntimeError("Missing dependency 'numpy'. Install requirements.txt first.") from exc
    return np


def _load_sklearn():
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
            roc_auc_score,
        )
        from sklearn.utils.class_weight import compute_sample_weight
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Missing dependency 'scikit-learn'. Install requirements.txt first."
        ) from exc
    return (
        GradientBoostingClassifier,
        accuracy_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
        compute_sample_weight,
    )


def _encode_features(rows):
    pd = _load_pandas()
    matrix = pd.get_dummies(
        rows[["symbol", *FEATURE_COLUMNS]],
        columns=["symbol"],
        dtype=float,
    )
    return matrix


def _build_metrics(y_true, probabilities, decision_threshold: float):
    _, accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, _ = _load_sklearn()

    predictions = (probabilities >= decision_threshold).astype(int)
    metrics = {
        "decision_threshold": float(decision_threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "positive_rate": float(y_true.mean()),
        "predicted_positive_rate": float(predictions.mean()),
    }

    unique_classes = set(y_true.tolist())
    if len(unique_classes) == 2:
        metrics["auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["auc"] = None
    return metrics


def _split_indices(row_count: int, train_split: float, validation_split: float):
    train_end = int(row_count * train_split)
    validation_end = int(row_count * (train_split + validation_split))
    train_end = max(train_end, 1)
    validation_end = max(validation_end, train_end + 1)
    validation_end = min(validation_end, row_count - 1)
    if train_end >= validation_end or validation_end >= row_count:
        raise ValueError(
            "Invalid train/validation/test split. Adjust train_split and validation_split in ModelConfig."
        )
    return train_end, validation_end


def _feature_importance_rows(model, columns):
    importances = getattr(model, "feature_importances_", None)
    if importances is None:
        return []

    pairs = zip(columns, importances)
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    return [
        {"feature": feature, "importance": float(importance)}
        for feature, importance in ordered[:15]
    ]


def _fit_classifier(x_train, y_train):
    GradientBoostingClassifier, _, _, _, _, _, compute_sample_weight = _load_sklearn()
    model = GradientBoostingClassifier(random_state=42)
    sample_weight = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(x_train, y_train, sample_weight=sample_weight)
    return model


def _prediction_rows(dataset_slice, probabilities):
    rows = dataset_slice[["timestamp", "symbol", "close", "future_return", "next_bar_return", "target"]].copy()
    rows["probability_up"] = probabilities
    return rows


def _strip_equity_curve(backtest_result):
    return {key: value for key, value in backtest_result.items() if key != "equity_curve"}


def _walk_forward_windows(row_count: int, validation_split: float, walk_forward_splits: int):
    if walk_forward_splits <= 0:
        return []

    validation_rows = max(int(row_count * validation_split), 500)
    test_rows = validation_rows
    minimum_train_rows = max(int(row_count * 0.35), validation_rows * 2, 1000)
    available_rows = row_count - validation_rows - minimum_train_rows
    if available_rows < test_rows:
        return []

    actual_splits = min(walk_forward_splits, available_rows // test_rows)
    if actual_splits <= 0:
        return []

    initial_train_end = row_count - actual_splits * test_rows - validation_rows
    windows = []
    for fold_index in range(actual_splits):
        train_end = initial_train_end + fold_index * test_rows
        validation_end = train_end + validation_rows
        test_end = min(validation_end + test_rows, row_count)
        windows.append((train_end, validation_end, test_end))
    return windows


def _run_walk_forward_analysis(dataset, features, targets, model_config):
    np = _load_numpy()

    fold_summaries = []
    for fold_number, (train_end, validation_end, test_end) in enumerate(
        _walk_forward_windows(
            row_count=len(dataset),
            validation_split=model_config.validation_split,
            walk_forward_splits=model_config.walk_forward_splits,
        ),
        start=1,
    ):
        x_train = features.iloc[:train_end]
        y_train = targets.iloc[:train_end]
        x_validation = features.iloc[train_end:validation_end]
        y_validation = targets.iloc[train_end:validation_end]
        x_test = features.iloc[validation_end:test_end]
        y_test = targets.iloc[validation_end:test_end]

        model = _fit_classifier(x_train, y_train)
        validation_probabilities = model.predict_proba(x_validation)[:, 1]
        validation_predictions = _prediction_rows(
            dataset.iloc[train_end:validation_end],
            validation_probabilities,
        )
        threshold_selection = select_probability_thresholds(
            validation_rows=validation_predictions,
            transaction_cost_bps=model_config.transaction_cost_bps,
            threshold_hysteresis=model_config.threshold_hysteresis,
            max_active_positions=model_config.max_active_positions,
        )

        test_probabilities = model.predict_proba(x_test)[:, 1]
        test_predictions = _prediction_rows(
            dataset.iloc[validation_end:test_end],
            test_probabilities,
        )
        test_metrics = _build_metrics(
            y_true=y_test,
            probabilities=test_probabilities,
            decision_threshold=threshold_selection["entry_probability"],
        )
        test_backtest = simulate_probability_strategy(
            prediction_rows=test_predictions,
            entry_probability=threshold_selection["entry_probability"],
            exit_probability=threshold_selection["exit_probability"],
            transaction_cost_bps=model_config.transaction_cost_bps,
            max_active_positions=model_config.max_active_positions,
        )
        fold_summaries.append(
            {
                "fold": fold_number,
                "train_rows": int(len(x_train)),
                "validation_rows": int(len(x_validation)),
                "test_rows": int(len(x_test)),
                "entry_probability": float(threshold_selection["entry_probability"]),
                "exit_probability": float(threshold_selection["exit_probability"]),
                "auc": test_metrics["auc"],
                "precision": test_metrics["precision"],
                "recall": test_metrics["recall"],
                "f1": test_metrics["f1"],
                "predicted_positive_rate": test_metrics["predicted_positive_rate"],
                "total_return": float(test_backtest["total_return"]),
                "sharpe": float(test_backtest["sharpe"]),
                "max_drawdown": float(test_backtest["max_drawdown"]),
                "trade_count": int(test_backtest["trade_count"]),
            }
        )

    if not fold_summaries:
        return {
            "summary": {
                "fold_count": 0,
                "profitable_folds": 0,
                "mean_auc": None,
                "mean_f1": None,
                "mean_total_return": None,
                "mean_sharpe": None,
                "worst_max_drawdown": None,
            },
            "folds": [],
        }

    auc_values = [row["auc"] for row in fold_summaries if row["auc"] is not None]
    summary = {
        "fold_count": len(fold_summaries),
        "profitable_folds": int(sum(row["total_return"] > 0.0 for row in fold_summaries)),
        "mean_auc": float(np.mean(auc_values)) if auc_values else None,
        "mean_f1": float(np.mean([row["f1"] for row in fold_summaries])),
        "mean_total_return": float(np.mean([row["total_return"] for row in fold_summaries])),
        "mean_sharpe": float(np.mean([row["sharpe"] for row in fold_summaries])),
        "worst_max_drawdown": float(min(row["max_drawdown"] for row in fold_summaries)),
    }
    return {
        "summary": summary,
        "folds": fold_summaries,
    }


def train_model_from_frames(frames: Mapping[str, object], model_config):
    pd = _load_pandas()
    joblib = _load_joblib()

    training_rows = []
    for symbol, frame in frames.items():
        rows = build_labeled_rows(
            symbol=symbol,
            price_frame=frame,
            horizon_bars=model_config.horizon_bars,
            positive_return_threshold=model_config.positive_return_threshold,
        )
        training_rows.append(rows)

    if not training_rows:
        raise ValueError("No training data was collected from IBKR.")

    dataset = pd.concat(training_rows, ignore_index=True)
    dataset = dataset.sort_values("timestamp").reset_index(drop=True)
    if len(dataset) < 100:
        raise ValueError("Dataset is too small. Increase duration or number of symbols.")

    features = _encode_features(dataset)
    targets = dataset["target"].astype(int)

    train_end, validation_end = _split_indices(
        row_count=len(dataset),
        train_split=model_config.train_split,
        validation_split=model_config.validation_split,
    )

    x_train = features.iloc[:train_end]
    x_validation = features.iloc[train_end:validation_end]
    x_test = features.iloc[validation_end:]
    y_train = targets.iloc[:train_end]
    y_validation = targets.iloc[train_end:validation_end]
    y_test = targets.iloc[validation_end:]

    model = _fit_classifier(x_train, y_train)

    validation_probabilities = model.predict_proba(x_validation)[:, 1]
    test_probabilities = model.predict_proba(x_test)[:, 1]

    validation_predictions = _prediction_rows(
        dataset.iloc[train_end:validation_end],
        validation_probabilities,
    )

    if model_config.entry_probability is None or model_config.exit_probability is None:
        threshold_selection = select_probability_thresholds(
            validation_rows=validation_predictions,
            transaction_cost_bps=model_config.transaction_cost_bps,
            threshold_hysteresis=model_config.threshold_hysteresis,
            max_active_positions=model_config.max_active_positions,
        )
        model_config.entry_probability = threshold_selection["entry_probability"]
        model_config.exit_probability = threshold_selection["exit_probability"]
        validation_backtest = threshold_selection["validation_backtest"]
    else:
        validation_backtest = simulate_probability_strategy(
            prediction_rows=validation_predictions,
            entry_probability=model_config.entry_probability,
            exit_probability=model_config.exit_probability,
            transaction_cost_bps=model_config.transaction_cost_bps,
            max_active_positions=model_config.max_active_positions,
        )

    test_predictions = _prediction_rows(
        dataset.iloc[validation_end:],
        test_probabilities,
    )

    validation_metrics = _build_metrics(
        y_true=y_validation,
        probabilities=validation_probabilities,
        decision_threshold=float(model_config.entry_probability),
    )
    test_metrics = _build_metrics(
        y_true=y_test,
        probabilities=test_probabilities,
        decision_threshold=float(model_config.entry_probability),
    )
    validation_metrics["train_rows"] = int(len(x_train))
    validation_metrics["validation_rows"] = int(len(x_validation))
    test_metrics["test_rows"] = int(len(x_test))

    test_backtest = simulate_probability_strategy(
        prediction_rows=test_predictions,
        entry_probability=float(model_config.entry_probability),
        exit_probability=float(model_config.exit_probability),
        transaction_cost_bps=model_config.transaction_cost_bps,
        max_active_positions=model_config.max_active_positions,
    )
    walk_forward_analysis = _run_walk_forward_analysis(dataset, features, targets, model_config)

    model_path = Path(model_config.model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "model": model,
        "feature_columns": list(x_train.columns),
        "model_config": asdict(model_config),
        "metrics": test_metrics,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "thresholds": {
            "entry_probability": float(model_config.entry_probability),
            "exit_probability": float(model_config.exit_probability),
        },
        "validation_backtest": _strip_equity_curve(validation_backtest),
        "test_backtest": _strip_equity_curve(test_backtest),
        "walk_forward_summary": walk_forward_analysis["summary"],
        "walk_forward_folds": walk_forward_analysis["folds"],
        "validation_predictions": validation_predictions.reset_index(drop=True),
        "test_predictions": test_predictions.reset_index(drop=True),
        "feature_importances": _feature_importance_rows(model, x_train.columns),
        "symbols": sorted(frames.keys()),
    }
    joblib.dump(bundle, model_path)
    return bundle


def load_model_bundle(model_path):
    joblib = _load_joblib()
    return joblib.load(model_path)


def predict_probability(bundle, symbol: str, price_frame):
    latest = build_latest_feature_row(symbol, price_frame)
    features = _encode_features(latest)
    features = features.reindex(columns=bundle["feature_columns"], fill_value=0.0)

    probability_up = float(bundle["model"].predict_proba(features)[0, 1])
    close = float(latest["close"].iloc[0])
    timestamp = latest["timestamp"].iloc[0]
    return {
        "symbol": symbol,
        "timestamp": str(timestamp),
        "close": close,
        "probability_up": probability_up,
    }

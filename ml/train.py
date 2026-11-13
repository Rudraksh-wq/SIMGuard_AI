"""Train the SIM-swap LightGBM and Isolation Forest scoring models."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split

from agents.features_text import FEATURE_COLUMNS, LABEL_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = PROJECT_ROOT / "sim_swap_fraud_dataset_100000.csv"
DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
HIGH_TIER_THRESHOLD = 70.0
ANOMALY_PERCENTILES = (5.0, 95.0)


def _read_dataset(dataset_path: Path) -> pd.DataFrame:
    data = pd.read_csv(dataset_path)
    required = set(FEATURE_COLUMNS) | {LABEL_COLUMN}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"dataset is missing columns: {', '.join(sorted(missing))}")

    data = data.loc[:, [*FEATURE_COLUMNS, LABEL_COLUMN]].copy()
    data = data.apply(pd.to_numeric, errors="raise")
    if not np.isfinite(data.to_numpy(dtype=float)).all():
        raise ValueError("dataset contains non-finite values")

    labels = set(data[LABEL_COLUMN].unique())
    if not labels.issubset({0, 1}) or len(labels) != 2:
        raise ValueError(f"{LABEL_COLUMN} must contain both binary labels 0 and 1")
    return data


def _json_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"unsupported JSON value: {type(value).__name__}")


def train_model(
    dataset_path: str | Path = DEFAULT_DATASET,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train, evaluate, and persist the scoring models and test metrics."""
    data = _read_dataset(Path(dataset_path))
    features = data.loc[:, FEATURE_COLUMNS]
    labels = data[LABEL_COLUMN].astype(int)

    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=random_state,
    )

    positive_count = int((y_train == 1).sum())
    negative_count = int((y_train == 0).sum())
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        scale_pos_weight=negative_count / positive_count,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
    )
    model.fit(x_train, y_train)

    normal_training_features = x_train.loc[y_train == 0]
    iforest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    iforest.fit(normal_training_features)

    training_normal_scores = iforest.decision_function(normal_training_features)
    lower_value, upper_value = np.percentile(training_normal_scores, ANOMALY_PERCENTILES)
    if not (math.isfinite(lower_value) and math.isfinite(upper_value) and upper_value > lower_value):
        raise ValueError("could not derive a finite anomaly scaling range from training data")

    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_raw_anomaly_scores = iforest.decision_function(x_test)
    test_anomaly = np.clip(
        (upper_value - test_raw_anomaly_scores) / (upper_value - lower_value),
        0.0,
        1.0,
    )
    test_risk_scores = 100.0 * (0.7 * test_probabilities + 0.3 * test_anomaly)
    test_predictions = (test_risk_scores >= HIGH_TIER_THRESHOLD).astype(int)

    metrics = {
        "data": "synthetic",
        "threshold": HIGH_TIER_THRESHOLD,
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, test_risk_scores)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
        },
        "split_sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
    }

    artifact_path = Path(artifact_dir)
    artifact_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path / "model.joblib")
    joblib.dump(iforest, artifact_path / "iforest.joblib")
    scaler = {
        "method": "training normal Isolation Forest decision_function percentiles",
        "lower_percentile": ANOMALY_PERCENTILES[0],
        "upper_percentile": ANOMALY_PERCENTILES[1],
        "lower_value": float(lower_value),
        "upper_value": float(upper_value),
        "direction": "lower decision_function values are more anomalous",
    }
    (artifact_path / "scaler.json").write_text(
        json.dumps(scaler, indent=2) + "\n",
        encoding="utf-8",
    )
    (artifact_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return metrics


def continue_model(
    dataset_path: str | Path = DEFAULT_DATASET,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
    random_state: int = 42,
    additional_rounds: int = 300,
    early_stopping_rounds: int = 50,
) -> dict[str, Any]:
    """Continue a saved LightGBM model and retain it only if validation PR-AUC improves."""
    if additional_rounds < 1:
        raise ValueError("additional_rounds must be at least 1")
    if early_stopping_rounds < 1:
        raise ValueError("early_stopping_rounds must be at least 1")

    data = _read_dataset(Path(dataset_path))
    features = data.loc[:, FEATURE_COLUMNS]
    labels = data[LABEL_COLUMN].astype(int)
    x_train, x_remaining, y_train, y_remaining = train_test_split(
        features,
        labels,
        test_size=0.30,
        stratify=labels,
        random_state=random_state,
    )
    x_validation, x_test, y_validation, y_test = train_test_split(
        x_remaining,
        y_remaining,
        test_size=0.50,
        stratify=y_remaining,
        random_state=random_state,
    )

    artifact_path = Path(artifact_dir)
    model_path = artifact_path / "model.joblib"
    iforest_path = artifact_path / "iforest.joblib"
    scaler_path = artifact_path / "scaler.json"
    missing = [path.name for path in (model_path, iforest_path, scaler_path) if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"cannot continue training; missing artifacts in {artifact_path}: "
            f"{', '.join(missing)}"
        )

    existing_model = joblib.load(model_path)
    if existing_model.booster_.feature_name() != list(FEATURE_COLUMNS):
        raise ValueError("saved LightGBM model features do not match the current feature schema")
    if existing_model.n_features_in_ != len(FEATURE_COLUMNS):
        raise ValueError("saved LightGBM model has an unexpected number of features")

    iforest = joblib.load(iforest_path)
    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
    lower_value = float(scaler["lower_value"])
    upper_value = float(scaler["upper_value"])
    if not (math.isfinite(lower_value) and math.isfinite(upper_value) and upper_value > lower_value):
        raise ValueError(f"invalid anomaly scaling range in {scaler_path}")

    validation_raw_anomaly = iforest.decision_function(x_validation)
    validation_anomaly = np.clip(
        (upper_value - validation_raw_anomaly) / (upper_value - lower_value),
        0.0,
        1.0,
    )

    def validation_pr_auc(candidate: LGBMClassifier) -> float:
        probabilities = candidate.predict_proba(x_validation)[:, 1]
        risk = 100.0 * (0.7 * probabilities + 0.3 * validation_anomaly)
        return float(average_precision_score(y_validation, risk))

    previous_rounds = int(existing_model.booster_.current_iteration())
    previous_validation_pr_auc = validation_pr_auc(existing_model)

    candidate = copy.deepcopy(existing_model)
    candidate.set_params(n_estimators=additional_rounds)
    candidate.fit(
        x_train,
        y_train,
        init_model=existing_model.booster_,
        eval_set=[(x_validation, y_validation)],
        callbacks=[
            early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            log_evaluation(period=0),
        ],
    )
    candidate_rounds = int(candidate.booster_.current_iteration())
    candidate_validation_pr_auc = validation_pr_auc(candidate)
    keep_candidate = candidate_validation_pr_auc > previous_validation_pr_auc
    model = candidate if keep_candidate else existing_model

    test_probabilities = model.predict_proba(x_test)[:, 1]
    test_raw_anomaly = iforest.decision_function(x_test)
    test_anomaly = np.clip(
        (upper_value - test_raw_anomaly) / (upper_value - lower_value),
        0.0,
        1.0,
    )
    test_risk_scores = 100.0 * (0.7 * test_probabilities + 0.3 * test_anomaly)
    test_predictions = (test_risk_scores >= HIGH_TIER_THRESHOLD).astype(int)

    metrics = {
        "data": "synthetic",
        "threshold": HIGH_TIER_THRESHOLD,
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "pr_auc": float(average_precision_score(y_test, test_risk_scores)),
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": confusion_matrix(y_test, test_predictions, labels=[0, 1]).tolist(),
        },
        "split_sizes": {
            "train": int(len(x_train)),
            "validation": int(len(x_validation)),
            "test": int(len(x_test)),
        },
        "continuation": {
            "initial_rounds": previous_rounds,
            "additional_rounds_requested": additional_rounds,
            "candidate_rounds": candidate_rounds,
            "validation_pr_auc_before": previous_validation_pr_auc,
            "validation_pr_auc_after": candidate_validation_pr_auc,
            "candidate_kept": keep_candidate,
            "selected_rounds": int(model.booster_.current_iteration()),
            "isolation_forest": "unchanged",
        },
    }

    if keep_candidate:
        joblib.dump(candidate, model_path)
    (artifact_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=_json_value) + "\n",
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--continue-existing",
        action="store_true",
        help="continue the saved LightGBM model instead of training from scratch",
    )
    parser.add_argument("--additional-rounds", type=int, default=300)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    args = parser.parse_args()
    if args.continue_existing:
        metrics = continue_model(
            args.data,
            args.artifacts,
            args.seed,
            args.additional_rounds,
            args.early_stopping_rounds,
        )
    else:
        metrics = train_model(args.data, args.artifacts, args.seed)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()

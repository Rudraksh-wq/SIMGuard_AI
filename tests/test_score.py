import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from agents.features_text import FEATURE_COLUMNS, LABEL_COLUMN
from ml.score import score_event
from ml.train import continue_model, train_model


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    root = Path(__file__).resolve().parents[1]
    artifacts = tmp_path_factory.mktemp("ml-artifacts")
    train_model(root / "sim_swap_fraud_dataset_100000.csv", artifacts)
    return artifacts


def test_training_persists_metrics_scaler_and_uses_stratified_splits(trained_artifacts):
    metrics = json.loads((trained_artifacts / "metrics.json").read_text(encoding="utf-8"))
    scaler = json.loads((trained_artifacts / "scaler.json").read_text(encoding="utf-8"))

    assert metrics["data"] == "synthetic"
    assert metrics["threshold"] == 70
    assert metrics["split_sizes"] == {"train": 70000, "validation": 15000, "test": 15000}
    assert len(metrics["confusion_matrix"]["matrix"]) == 2
    assert 0 <= scaler["lower_percentile"] < scaler["upper_percentile"] <= 100
    assert (trained_artifacts / "model.joblib").is_file()
    assert (trained_artifacts / "iforest.joblib").is_file()


def test_fraud_like_and_normal_events_score_in_expected_tiers(trained_artifacts):
    dataset = pd.read_csv(Path(__file__).resolve().parents[1] / "sim_swap_fraud_dataset_100000.csv")
    fraud = dataset.loc[dataset[LABEL_COLUMN] == 1, FEATURE_COLUMNS].iloc[0].to_dict()
    normal = dataset.loc[dataset[LABEL_COLUMN] == 0, FEATURE_COLUMNS].iloc[0].to_dict()

    fraud_result = score_event(fraud, trained_artifacts)
    normal_result = score_event(normal, trained_artifacts)

    assert fraud_result.risk_score >= 70
    assert normal_result.risk_score < 40
    assert 0 <= fraud_result.risk_score <= 100
    assert 0 <= normal_result.risk_score <= 100
    assert len(fraud_result.contributions) == 5
    assert all(feature in FEATURE_COLUMNS for feature in fraud_result.contributions)


def test_continuation_adds_rounds_without_retraining_isolation_forest(trained_artifacts):
    root = Path(__file__).resolve().parents[1]
    dataset = pd.read_csv(root / "sim_swap_fraud_dataset_100000.csv")
    normal_rows = dataset.loc[dataset[LABEL_COLUMN] == 0, FEATURE_COLUMNS].iloc[:5]
    forest_before = joblib.load(trained_artifacts / "iforest.joblib")
    predictions_before = forest_before.decision_function(normal_rows)

    metrics = continue_model(
        root / "sim_swap_fraud_dataset_100000.csv",
        trained_artifacts,
        additional_rounds=3,
        early_stopping_rounds=2,
    )

    continuation = metrics["continuation"]
    forest_after = joblib.load(trained_artifacts / "iforest.joblib")
    predictions_after = forest_after.decision_function(normal_rows)
    initial = continuation["initial_rounds"]
    assert 0 < initial < 1000  # early stopping trimmed the 1000-round budget
    assert initial < continuation["candidate_rounds"] <= initial + 3
    assert continuation["isolation_forest"] == "unchanged"
    assert np.allclose(predictions_before, predictions_after)


def test_compiled_isolation_forest_matches_sklearn(trained_artifacts):
    from ml.score import _load_artifacts

    dataset = pd.read_csv(Path(__file__).resolve().parents[1] / "sim_swap_fraud_dataset_100000.csv")
    rows = dataset.loc[:, FEATURE_COLUMNS].iloc[:3000].to_numpy(dtype=float)
    forest = joblib.load(trained_artifacts / "iforest.joblib")
    compiled = _load_artifacts(trained_artifacts).forest
    assert np.allclose(compiled.decision_function(rows), forest.decision_function(rows), atol=1e-12)


def test_batch_scoring_equals_single_scoring(trained_artifacts):
    from ml.score import score_batch

    dataset = pd.read_csv(Path(__file__).resolve().parents[1] / "sim_swap_fraud_dataset_100000.csv")
    events = dataset.loc[:, FEATURE_COLUMNS].iloc[:20].to_dict("records")
    batch = score_batch(events, trained_artifacts)
    for event, result in zip(events, batch):
        single = score_event(event, trained_artifacts)
        assert abs(single.risk_score - result.risk_score) < 1e-9
        assert single.contributions == pytest.approx(result.contributions)

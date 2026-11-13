"""Load trained scoring artifacts and score one SIM-swap event."""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from agents.features_text import FEATURE_COLUMNS, _read_feature_values
from agents.scheme import MLResult


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
HIGH_TIER_THRESHOLD = 70.0
MEDIUM_TIER_THRESHOLD = 40.0
_ARTIFACT_CACHE: dict[Path, tuple[Any, Any, dict[str, Any]]] = {}


def _load_artifacts(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> tuple[Any, Any, dict[str, Any]]:
    path = Path(artifact_dir).resolve()
    if path not in _ARTIFACT_CACHE:
        model_path = path / "model.joblib"
        iforest_path = path / "iforest.joblib"
        scaler_path = path / "scaler.json"
        missing = [file.name for file in (model_path, iforest_path, scaler_path) if not file.is_file()]
        if missing:
            raise FileNotFoundError(
                f"scoring artifacts are missing from {path}: {', '.join(missing)}; "
                "run ml/train.py first"
            )
        model = joblib.load(model_path)
        iforest = joblib.load(iforest_path)
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        lower = float(scaler["lower_value"])
        upper = float(scaler["upper_value"])
        if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
            raise ValueError(f"invalid anomaly scaling range in {scaler_path}")
        _ARTIFACT_CACHE[path] = (model, iforest, scaler)
    return _ARTIFACT_CACHE[path]


def score_event(
    event: Mapping[str, object],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> MLResult:
    """Return a 0-100 risk score and the five strongest LightGBM contributions."""
    started = time.perf_counter()
    values = _read_feature_values(event)
    row = pd.DataFrame([[values[column] for column in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
    model, iforest, scaler = _load_artifacts(artifact_dir)

    probability = float(model.predict_proba(row)[0, 1])
    raw_anomaly_score = float(iforest.decision_function(row)[0])
    anomaly = float(
        np.clip(
            (float(scaler["upper_value"]) - raw_anomaly_score)
            / (float(scaler["upper_value"]) - float(scaler["lower_value"])),
            0.0,
            1.0,
        )
    )
    risk_score = float(np.clip(100.0 * (0.7 * probability + 0.3 * anomaly), 0.0, 100.0))

    raw_contributions = np.asarray(model.predict(row, pred_contrib=True))
    if raw_contributions.ndim == 2:
        raw_contributions = raw_contributions[0]
    if raw_contributions.ndim != 1 or len(raw_contributions) != len(FEATURE_COLUMNS) + 1:
        raise ValueError("LightGBM returned an unexpected pred_contrib shape")
    feature_contributions = [
        (column, float(value))
        for column, value in zip(FEATURE_COLUMNS, raw_contributions[:-1])
    ]
    top_contributions = sorted(feature_contributions, key=lambda item: abs(item[1]), reverse=True)[:5]

    if risk_score >= HIGH_TIER_THRESHOLD:
        tier = "high"
    elif risk_score >= MEDIUM_TIER_THRESHOLD:
        tier = "medium"
    else:
        tier = "low"
    return MLResult(
        risk_score=risk_score,
        tier=tier,
        lgbm_prob=probability,
        anomaly=anomaly,
        contributions=dict(top_contributions),
        latency_ms=(time.perf_counter() - started) * 1000.0,
    )

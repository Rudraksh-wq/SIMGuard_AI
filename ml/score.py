"""Load trained scoring artifacts and score SIM-swap events (single or batched).

Speed notes:
- LightGBM is called through its native Booster on float64 numpy arrays (no DataFrame, no
  sklearn wrapper), which is ~10x faster for one row.
- The Isolation Forest is "compiled" once at load time into flat numpy arrays and evaluated for
  all trees and all rows at once. sklearn walks its 200 trees one by one in Python (~17 ms per
  row); the compiled version gives the same decision_function (checked at load time) in well
  under a millisecond.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from agents.features_text import FEATURE_COLUMNS, _read_feature_values
from agents.schemas import MLResult


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parent / "artifacts"
HIGH_TIER_THRESHOLD = 70.0
MEDIUM_TIER_THRESHOLD = 40.0
LGBM_WEIGHT = 0.7
ANOMALY_WEIGHT = 0.3
TOP_CONTRIBUTIONS = 5


def _average_path_length(n: np.ndarray) -> np.ndarray:
    """sklearn's c(n): expected path length of an unsuccessful BST search."""
    n = np.asarray(n, dtype=np.float64)
    out = np.zeros_like(n)
    out[n == 2] = 1.0
    big = n > 2
    out[big] = 2.0 * (np.log(n[big] - 1.0) + np.euler_gamma) - 2.0 * (n[big] - 1.0) / n[big]
    return out


class CompiledIsolationForest:
    """Vectorised re-implementation of IsolationForest.decision_function."""

    def __init__(self, forest: Any) -> None:
        trees = forest.estimators_
        features = forest.estimators_features_
        n_trees = len(trees)
        max_nodes = max(tree.tree_.node_count for tree in trees)
        self.left = np.zeros((n_trees, max_nodes), dtype=np.int64)
        self.right = np.zeros((n_trees, max_nodes), dtype=np.int64)
        self.feature = np.zeros((n_trees, max_nodes), dtype=np.int64)
        self.threshold = np.zeros((n_trees, max_nodes), dtype=np.float64)
        self.leaf_value = np.zeros((n_trees, max_nodes), dtype=np.float64)
        max_depth = 0
        for t, (tree, used) in enumerate(zip(trees, features)):
            structure = tree.tree_
            count = structure.node_count
            left = structure.children_left
            right = structure.children_right
            is_leaf = left == -1
            depth = np.zeros(count, dtype=np.int64)
            for node in range(count):  # parents always precede children in sklearn trees
                if not is_leaf[node]:
                    depth[left[node]] = depth[node] + 1
                    depth[right[node]] = depth[node] + 1
            max_depth = max(max_depth, int(depth.max()))
            nodes = np.arange(count)
            # leaves point to themselves, so extra traversal steps are no-ops
            self.left[t, :count] = np.where(is_leaf, nodes, left)
            self.right[t, :count] = np.where(is_leaf, nodes, right)
            used = np.asarray(used)
            self.feature[t, :count] = np.where(is_leaf, 0, used[np.maximum(structure.feature, 0)])
            self.threshold[t, :count] = structure.threshold
            self.leaf_value[t, :count] = np.where(
                is_leaf,
                # sklearn: (nodes on path = depth + 1) + c(n_leaf) - 1
                depth + _average_path_length(structure.n_node_samples),
                0.0,
            )
        self.steps = max_depth
        self.tree_index = np.arange(n_trees)[:, None]
        self.denominator = n_trees * float(_average_path_length(np.array([forest.max_samples_]))[0])
        self.offset = float(forest.offset_)

    def decision_function(self, x: np.ndarray) -> np.ndarray:
        # sklearn trees compare float32-cast inputs against their thresholds
        x = np.asarray(x, dtype=np.float32).astype(np.float64)
        n_rows = x.shape[0]
        rows = np.arange(n_rows)[None, :]
        node = np.zeros((len(self.left), n_rows), dtype=np.int64)
        for _ in range(self.steps):
            feature = self.feature[self.tree_index, node]
            go_left = x[rows, feature] <= self.threshold[self.tree_index, node]
            node = np.where(go_left, self.left[self.tree_index, node], self.right[self.tree_index, node])
        depths = self.leaf_value[self.tree_index, node].sum(axis=0)
        scores = -(2.0 ** (-depths / self.denominator))
        return scores - self.offset


@dataclass
class ScoringArtifacts:
    booster: Any
    forest: CompiledIsolationForest
    lower: float
    upper: float
    metrics: dict[str, Any]


_ARTIFACT_CACHE: dict[Path, ScoringArtifacts] = {}


def _load_artifacts(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> ScoringArtifacts:
    path = Path(artifact_dir).resolve()
    if path not in _ARTIFACT_CACHE:
        model_path = path / "model.joblib"
        iforest_path = path / "iforest.joblib"
        scaler_path = path / "scaler.json"
        missing = [file.name for file in (model_path, iforest_path, scaler_path) if not file.is_file()]
        if missing:
            raise FileNotFoundError(
                f"scoring artifacts are missing from {path}: {', '.join(missing)}; "
                "run `python -m ml.train` first"
            )
        model = joblib.load(model_path)
        iforest = joblib.load(iforest_path)
        scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
        lower = float(scaler["lower_value"])
        upper = float(scaler["upper_value"])
        if not (math.isfinite(lower) and math.isfinite(upper) and upper > lower):
            raise ValueError(f"invalid anomaly scaling range in {scaler_path}")

        compiled = CompiledIsolationForest(iforest)
        probe = np.random.default_rng(0).normal(size=(64, len(FEATURE_COLUMNS))) * 50 + 50
        if not np.allclose(compiled.decision_function(probe), iforest.decision_function(probe), atol=1e-9):
            raise RuntimeError("compiled Isolation Forest disagrees with sklearn; refusing to score")

        metrics_path = path / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
        _ARTIFACT_CACHE[path] = ScoringArtifacts(model.booster_, compiled, lower, upper, metrics)
    return _ARTIFACT_CACHE[path]


def _tier(score: float) -> str:
    if score >= HIGH_TIER_THRESHOLD:
        return "high"
    if score >= MEDIUM_TIER_THRESHOLD:
        return "medium"
    return "low"


def score_batch(
    events: Sequence[Mapping[str, object]],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> list[MLResult]:
    """Score many events in one vectorised pass. latency_ms is the per-event share."""
    if not events:
        return []
    started = time.perf_counter()
    matrix = np.array(
        [[_read_feature_values(event)[column] for column in FEATURE_COLUMNS] for event in events],
        dtype=np.float64,
    )
    artifacts = _load_artifacts(artifact_dir)

    probabilities = np.asarray(artifacts.booster.predict(matrix), dtype=np.float64)
    raw_anomaly = artifacts.forest.decision_function(matrix)
    anomaly = np.clip((artifacts.upper - raw_anomaly) / (artifacts.upper - artifacts.lower), 0.0, 1.0)
    risk = np.clip(100.0 * (LGBM_WEIGHT * probabilities + ANOMALY_WEIGHT * anomaly), 0.0, 100.0)

    contributions = np.asarray(artifacts.booster.predict(matrix, pred_contrib=True))
    if contributions.shape != (len(events), len(FEATURE_COLUMNS) + 1):
        raise ValueError("LightGBM returned an unexpected pred_contrib shape")

    per_event_ms = (time.perf_counter() - started) * 1000.0 / len(events)
    results = []
    for index in range(len(events)):
        row = contributions[index, :-1]
        top = np.argsort(-np.abs(row))[:TOP_CONTRIBUTIONS]
        score = float(risk[index])
        results.append(
            MLResult(
                risk_score=score,
                tier=_tier(score),
                lgbm_prob=float(probabilities[index]),
                anomaly=float(anomaly[index]),
                contributions={FEATURE_COLUMNS[i]: float(row[i]) for i in top},
                latency_ms=per_event_ms,
            )
        )
    return results


def score_event(
    event: Mapping[str, object],
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> MLResult:
    """Return a 0-100 risk score and the five strongest LightGBM contributions."""
    return score_batch([event], artifact_dir)[0]


def model_info(artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> dict[str, Any]:
    artifacts = _load_artifacts(artifact_dir)
    return {
        "lgbm_trees": int(artifacts.booster.current_iteration()),
        "iforest_trees": int(len(artifacts.forest.left)),
        "metrics": artifacts.metrics,
    }

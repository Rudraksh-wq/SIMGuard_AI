"""Agent 1: the fine-tuned Laya decision model (non-generative, returns p_fraud only).

The checkpoint is loaded once, lazily, and shared across requests. GPU is used when available.
The gate (block / allow / human_review) uses the cut-offs calibrated during fine-tuning
(final/gate.json); between them the case goes to a human analyst.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from agents.features_text import FEATURE_COLUMNS, LAYA_QUESTION, LAYA_QUESTION_ID, laya_text
from agents.schemas import LayaResult

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "finetune" / "fraud_v2" / "runs" / "20261113_171802_a1_6ep" / "final"
)

_lock = threading.Lock()
_agent: Any = None
_load_error: str | None = None
_gate: dict[str, float] = {"lo": 0.12, "hi": 0.88}
CHECKPOINT = ""  # filled in after load; read by api.pipeline.laya_checkpoint()


def checkpoint_path() -> Path:
    load_dotenv()
    return Path(os.getenv("LAYA_MODEL") or DEFAULT_CHECKPOINT)


def _device() -> str:
    wanted = os.getenv("LAYA_DEVICE", "auto").lower()
    if wanted != "auto":
        return wanted
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _ensure_loaded() -> Any:
    global _agent, _load_error, CHECKPOINT, _gate
    if _agent is not None:
        return _agent
    with _lock:
        if _agent is not None:
            return _agent
        path = checkpoint_path()
        try:
            import laya

            if not path.exists():
                raise FileNotFoundError(f"Laya checkpoint not found at {path}")
            gate_file = path / "gate.json"
            if gate_file.is_file():
                gate = json.loads(gate_file.read_text(encoding="utf-8"))
                _gate = {"lo": float(gate["lo"]), "hi": float(gate["hi"])}
            agent = laya.load(str(path), device=_device())
            # one dummy forward pass so the first real request doesn't pay CUDA warm-up (~0.5 s)
            agent.predict(
                laya_text({c: 0 for c in FEATURE_COLUMNS}), {LAYA_QUESTION_ID: dict(LAYA_QUESTION)}
            )
            _agent = agent
            CHECKPOINT = f"{path.parent.name}/{path.name}"
            _load_error = None
        except Exception as error:  # keep the API alive; the pipeline falls back to ML only
            _load_error = f"{type(error).__name__}: {error}"
            raise
    return _agent


def gate_for(p_fraud: float) -> str:
    if p_fraud >= _gate["hi"]:
        return "block"
    if p_fraud <= _gate["lo"]:
        return "allow"
    return "human_review"


def decide_batch(events: Sequence[Mapping[str, object]], batch_size: int = 64) -> list[LayaResult]:
    if not events:
        return []
    started = time.perf_counter()
    try:
        agent = _ensure_loaded()
        texts = [laya_text(event) for event in events]
        questions = {LAYA_QUESTION_ID: dict(LAYA_QUESTION)}
        with _lock:  # one forward pass at a time on the shared model
            if len(texts) == 1:
                answers = [agent.predict(texts[0], questions)]
            else:
                answers = agent.predict_batch(texts, questions, batch_size=batch_size)
        probabilities = [float(item["answers"][LAYA_QUESTION_ID]["noul"]) for item in answers]
    except Exception as error:
        per_event = (time.perf_counter() - started) * 1000.0 / len(events)
        message = _load_error or f"{type(error).__name__}: {error}"
        return [
            LayaResult(p_fraud=None, model="unavailable", latency_ms=per_event, error=message)
            for _ in events
        ]
    per_event = (time.perf_counter() - started) * 1000.0 / len(events)
    return [
        LayaResult(p_fraud=p, model=CHECKPOINT, latency_ms=per_event, gate=gate_for(p))
        for p in probabilities
    ]


def decide(event: Mapping[str, object]) -> LayaResult:
    return decide_batch([event])[0]


def status() -> dict[str, Any]:
    return {
        "loaded": _agent is not None,
        "checkpoint": CHECKPOINT or str(checkpoint_path()),
        "gate": dict(_gate),
        "error": _load_error,
    }

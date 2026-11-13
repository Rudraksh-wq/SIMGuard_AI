import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

from agents.schemas import Event, LayaResult, MLResult


@dataclass(frozen=True)
class RuntimeConfig:
    laya_fraud_threshold: float
    laya_weight: float
    mock_models: bool
    ollama_url: str
    qwen_model: str
    laya_model: str
    laya_base_model: str


def get_runtime_config() -> RuntimeConfig:
    load_dotenv()

    threshold_raw = os.getenv("LAYA_FRAUD_THRESHOLD")
    try:
        threshold = float(threshold_raw) if threshold_raw is not None else 0.5
    except ValueError as error:
        raise ValueError("LAYA_FRAUD_THRESHOLD must be a float") from error
    if not 0 < threshold < 1:
        raise ValueError("LAYA_FRAUD_THRESHOLD must be strictly between 0 and 1")

    try:
        weight = float(os.getenv("LAYA_WEIGHT", "0.7"))
    except ValueError as error:
        raise ValueError("LAYA_WEIGHT must be a float between 0 and 1") from error
    if not 0 <= weight <= 1:
        raise ValueError("LAYA_WEIGHT must be between 0 and 1")

    mock_value = os.getenv("MOCK_MODELS", "0").strip().lower()
    if mock_value not in {"0", "1", "false", "true", "no", "yes"}:
        raise ValueError("MOCK_MODELS must be 0 or 1")

    return RuntimeConfig(
        laya_fraud_threshold=threshold,
        laya_weight=weight,
        mock_models=mock_value in {"1", "true", "yes"},
        ollama_url=os.getenv("OLLAMA_URL", "http://localhost:11434").rstrip("/"),
        qwen_model=os.getenv("QWEN_MODEL", "qwen2.5:7b"),
        laya_model=os.getenv("LAYA_MODEL", ""),
        laya_base_model=os.getenv("LAYA_BASE_MODEL", ""),
    )


def is_fraud_like(event: Event) -> bool:
    return (
        event.imei_changed == 1
        and event.location_gap_km > 300
        and event.otp_burst_count >= 4
    )


def score_event(event: Event) -> MLResult:
    fraud_like = is_fraud_like(event)
    score = 90.0 if fraud_like else 20.0
    return MLResult(
        risk_score=score,
        tier="high" if fraud_like else "low",
        lgbm_prob=0.9 if fraud_like else 0.1,
        anomaly=0.9 if fraud_like else 0.1,
        contributions={},
        latency_ms=0.0,
    )


def decide(event: Event) -> LayaResult:
    fraud_like = is_fraud_like(event)
    return LayaResult(
        p_fraud=0.95 if fraud_like else 0.05,
        model="mock",
        latency_ms=0.0,
    )


def template_report(payload: dict[str, Any]) -> str:
    evidence = payload["evidence"]
    evidence_text = (
        "\n".join(f"- {indicator}" for indicator in evidence)
        if evidence
        else "- No evidence indicators were available."
    )
    return (
        f"Summary: This request has a {payload['tier']} risk score of "
        f"{payload['risk_score']:.1f}. A human analyst must decide.\n"
        f"Why it was flagged:\n{evidence_text}\n"
        "Recommended actions: Hold the request, notify the subscriber via a "
        "verified channel, and perform extra identity verification."
    )


def generate_report(payload: dict[str, Any]) -> tuple[str, str]:
    return template_report(payload), "template"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

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
        ollama_url=os.getenv("QWEN_ENDPOINT", "").rstrip("/"),
        qwen_model=os.getenv("QWEN_MODEL", "Qwen/Qwen3.5-2B"),
        laya_model=os.getenv("LAYA_MODEL", ""),
        laya_base_model=os.getenv("LAYA_BASE_MODEL", ""),
    )


def is_fraud_like(event: Event) -> bool:
    identity_changes = (
        event.device_change_flag + event.sim_type_change_flag
        + event.imsi_change_flag + event.iccid_change_flag
    )
    return identity_changes >= 2 and event.otp_and_sim_change_geo_hash_length < 0.5


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
    """Deterministic fallback report built only from the supplied payload."""
    p = payload.get("parameters", payload)
    evidence = p.get("evidence_indicators", p.get("evidence", []))
    evidence_text = (
        "\n".join(f"- {indicator}" for indicator in evidence)
        if evidence
        else "- No rule-based indicators fired."
    )
    score = p.get("final_risk_score", p.get("risk_score"))
    tier = p.get("final_tier", p.get("tier"))
    ml = p.get("lightgbm_isolation_forest", {})
    laya = p.get("laya", {})
    lines = [
        f"Summary: Verdict {payload.get('verdict', 'none')}. Risk score {score:.1f} ({tier}). "
        "This is suspected risk for human review, not confirmed fraud.",
        "Supplied evidence:",
        evidence_text,
    ]
    if ml:
        lines.append(
            f"- LightGBM fraud probability {ml.get('lightgbm_fraud_probability')}, "
            f"Isolation Forest anomaly {ml.get('isolation_forest_anomaly_0_to_1')}"
        )
    if laya:
        lines.append(f"- Laya p_fraud {laya.get('p_fraud')} (gate: {laya.get('gate')})")
    lines += [
        "Limitations: Scores come from models trained on synthetic data and do not prove intent.",
        "Suggested review steps: Hold the request for review, contact the subscriber through a "
        "verified channel, and perform extra identity verification.",
    ]
    return "\n".join(lines)


def generate_report(payload: dict[str, Any]) -> tuple[str, str]:
    return template_report(payload), "template"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

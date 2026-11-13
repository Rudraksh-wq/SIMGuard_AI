import importlib
import time
from collections.abc import Mapping
from typing import Any
from uuid import uuid4

from agents.schemas import AnalyzeResponse, Event, FinalScore, LayaResult, MLResult
from api import mocks
from api.db import save_case
from api.fusion import compute_final, tier_for

try:
    from ml.score import score_event as _live_score_event
except ModuleNotFoundError as error:
    if error.name not in {"ml", "ml.score"}:
        raise
    _live_score_event = None

try:
    from agents.laya_agent import decide as _live_decide
except ModuleNotFoundError as error:
    if error.name != "agents.laya_agent":
        raise
    _live_decide = None

try:
    from agents.evidence import extract_evidence as _live_extract_evidence
except ModuleNotFoundError as error:
    if error.name != "agents.evidence":
        raise
    _live_extract_evidence = None

try:
    from agents.qwen_agent import generate_report as _live_generate_report
except ModuleNotFoundError as error:
    if error.name != "agents.qwen_agent":
        raise
    _live_generate_report = None

_last_laya_checkpoint: str | None = None


def score_event(event: Event) -> MLResult:
    if mocks.get_runtime_config().mock_models:
        return mocks.score_event(event)
    if _live_score_event is None:
        raise RuntimeError("ml.score.score_event is unavailable; enable MOCK_MODELS=1")
    return _live_score_event(event)


def decide(event: Event) -> LayaResult:
    if mocks.get_runtime_config().mock_models:
        return mocks.decide(event)
    if _live_decide is None:
        raise RuntimeError("agents.laya_agent.decide is unavailable; enable MOCK_MODELS=1")
    return _live_decide(event)


def extract_evidence(event: Event) -> list[str]:
    if mocks.get_runtime_config().mock_models:
        return []
    if _live_extract_evidence is None:
        raise RuntimeError(
            "agents.evidence.extract_evidence is unavailable; enable MOCK_MODELS=1"
        )
    try:
        return _live_extract_evidence(event.model_dump())
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "The live evidence extractor is incompatible with agents.schemas.Event"
        ) from error


def generate_report(payload: dict[str, Any]) -> Any:
    if mocks.get_runtime_config().mock_models:
        return mocks.generate_report(payload)
    if _live_generate_report is None:
        raise RuntimeError(
            "agents.qwen_agent.generate_report is unavailable; enable MOCK_MODELS=1"
        )
    return _live_generate_report(payload)


def _report_parts(result: Any) -> tuple[str, str]:
    if isinstance(result, tuple) and len(result) == 2:
        report, source = result
    elif isinstance(result, Mapping):
        report = result.get("report")
        source = result.get("report_source", "qwen")
    elif isinstance(result, str):
        report, source = result, "qwen"
    else:
        raise TypeError("Qwen report function must return text or a report/source pair")

    if not isinstance(report, str) or source not in {"qwen", "template"}:
        raise ValueError("Qwen report function returned an invalid report or source")
    return report, source


def _final_score(
    event_laya: float | None,
    ml_risk: float,
    config: mocks.RuntimeConfig,
) -> FinalScore:
    score, z_laya, z_ml, weight = compute_final(
        event_laya, ml_risk, config.laya_fraud_threshold, config.laya_weight
    )
    rounded_score = round(score, 1)
    return FinalScore(
        score=rounded_score,
        tier=tier_for(rounded_score),
        laya_z=z_laya,
        ml_z=z_ml,
        laya_weight_used=weight,
    )


def _make_report(
    payload: dict[str, Any],
) -> tuple[str, str, float]:
    started = time.perf_counter()
    try:
        result = generate_report(payload)
        report, source = _report_parts(result)
    except Exception:
        report = mocks.template_report(payload)
        source = "template"
    latency_ms = (time.perf_counter() - started) * 1000
    return report, source, latency_ms


def analyze(event: Event) -> AnalyzeResponse:
    global _last_laya_checkpoint

    config = mocks.get_runtime_config()
    ml = score_event(event)
    laya = decide(event)
    _last_laya_checkpoint = laya.model or None
    evidence = extract_evidence(event)

    p_laya = laya.p_fraud
    final = _final_score(p_laya, ml.risk_score, config)
    laya_flag = p_laya is not None and p_laya >= config.laya_fraud_threshold
    ml_flag = ml.risk_score >= 70
    fraud_detected = laya_flag or ml_flag
    agreement = (
        "laya_unavailable"
        if p_laya is None
        else "agree"
        if laya_flag == ml_flag
        else "disagree"
    )

    report: str | None = None
    report_source = "none"
    report_latency_ms: float | None = None
    if fraud_detected:
        payload = {
            "risk_score": final.score,
            "tier": final.tier,
            "evidence": evidence,
            "laya_p_fraud": p_laya,
            "agreement": agreement,
        }
        report, report_source, report_latency_ms = _make_report(payload)

    response = AnalyzeResponse(
        case_id=uuid4().hex,
        event=event,
        ml=ml,
        laya=laya,
        final=final,
        evidence=evidence,
        fraud_detected=fraud_detected,
        agreement=agreement,
        report=report,
        report_source=report_source,
        report_latency_ms=report_latency_ms,
        created_at=mocks.utc_now_iso(),
    )
    save_case(response)
    return response


def laya_checkpoint() -> str:
    if _last_laya_checkpoint:
        return _last_laya_checkpoint
    if _live_decide is None:
        return "unknown"
    model_name = getattr(_live_decide, "last_model", None)
    if model_name:
        return str(model_name)
    module = importlib.import_module(_live_decide.__module__)
    for attribute in ("CHECKPOINT", "ACTIVE_MODEL", "LOADED_MODEL"):
        value = getattr(module, attribute, None)
        if value:
            return str(value)
    return "unknown"

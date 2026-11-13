"""Event -> LightGBM + Isolation Forest -> Laya -> evidence -> fused verdict -> Qwen report.

The cheap models (ML ~ms, Laya ~10-40 ms) see every event. Qwen (seconds, remote on Kaggle)
only writes reports for flagged / human-review cases, capped per request; any other case can
get a report on demand via POST /cases/{id}/report.
"""

from __future__ import annotations

import os
import queue
import threading
import time
from collections.abc import Sequence
from typing import Any
from uuid import uuid4

from agents.schemas import AnalyzeResponse, Event, FinalScore, InputInfo, LayaResult, MLResult
from api import mocks
from api.db import get_case, save_case, update_case
from api.fusion import compute_final, tier_for
from api.input_parser import strip_label

try:
    from ml.score import score_batch as _live_score_batch
except ModuleNotFoundError as error:
    if error.name not in {"ml", "ml.score"}:
        raise
    _live_score_batch = None

try:
    from agents import laya_agent as _laya_module
except ModuleNotFoundError as error:
    if error.name != "agents.laya_agent":
        raise
    _laya_module = None

from agents.evidence import extract_evidence as _live_extract_evidence
from agents import qwen_agent as _qwen_module

ML_HIGH = 70.0
MAX_RAW_FOR_QWEN = 1500


def _auto_report_limit() -> int:
    try:
        return max(0, int(os.getenv("QWEN_MAX_AUTO_REPORTS", "3")))
    except ValueError:
        return 3


# --- stage hooks (patched in tests, swapped for mocks in MOCK_MODELS mode) --------------------
def score_events(events: Sequence[Event]) -> list[MLResult]:
    if mocks.get_runtime_config().mock_models:
        return [mocks.score_event(event) for event in events]
    if _live_score_batch is None:
        raise RuntimeError("ml.score is unavailable; enable MOCK_MODELS=1")
    return _live_score_batch([event.features() for event in events])


def decide_events(events: Sequence[Event]) -> list[LayaResult]:
    if mocks.get_runtime_config().mock_models:
        return [mocks.decide(event) for event in events]
    if _laya_module is None:
        return [LayaResult(p_fraud=None, model="unavailable", latency_ms=0.0,
                           error="agents.laya_agent missing") for _ in events]
    return _laya_module.decide_batch([event.features() for event in events])


def extract_evidence(event: Event) -> list[str]:
    if mocks.get_runtime_config().mock_models:
        return []
    return _live_extract_evidence(event.features())


def generate_report(payload: dict[str, Any]) -> Any:
    if mocks.get_runtime_config().mock_models:
        return mocks.generate_report(payload)
    return _qwen_module.generate_report(payload)


# --- helpers ---------------------------------------------------------------------------------
def _report_parts(result: Any) -> tuple[str, str, str | None]:
    model = None
    if isinstance(result, tuple) and len(result) == 2:
        report, source = result
    elif isinstance(result, dict):
        report = result.get("report")
        source = result.get("report_source", "qwen")
        model = result.get("report_model")
    elif isinstance(result, str):
        report, source = result, "qwen"
    else:
        raise TypeError("Qwen report function must return text or a report/source pair")
    if not isinstance(report, str) or source not in {"qwen", "template"}:
        raise ValueError("Qwen report function returned an invalid report or source")
    return report, source, model


def _final_score(p_laya: float | None, ml_risk: float, config: mocks.RuntimeConfig) -> FinalScore:
    score, z_laya, z_ml, weight = compute_final(
        p_laya, ml_risk, config.laya_fraud_threshold, config.laya_weight
    )
    rounded = round(score, 1)
    return FinalScore(score=rounded, tier=tier_for(rounded), laya_z=z_laya, ml_z=z_ml,
                      laya_weight_used=weight)


def build_report_payload(response: AnalyzeResponse) -> dict[str, Any]:
    """The JSON the Kaggle Qwen service receives (verdict + parameters + input context)."""
    context, _label = strip_label(response.input.extra)
    laya = response.laya
    parameters: dict[str, Any] = {
        "context_note": (
            "All data is synthetic. The verdict comes from the detectors below; "
            "a human analyst makes the final decision."
        ),
        "final_risk_score": response.final.score,
        "final_tier": response.final.tier,
        "decision_rule": (
            f"flagged if Laya p_fraud >= {mocks.get_runtime_config().laya_fraud_threshold} "
            f"or ML risk score >= {ML_HIGH:g}"
        ),
        "lightgbm_isolation_forest": {
            "ml_risk_score": round(response.ml.risk_score, 1),
            "lightgbm_fraud_probability": round(response.ml.lgbm_prob, 4),
            "isolation_forest_anomaly_0_to_1": round(response.ml.anomaly, 4),
            "top_feature_contributions_log_odds": {
                k: round(v, 3) for k, v in response.ml.contributions.items()
            },
        },
        "laya": {
            "p_fraud": None if laya.p_fraud is None else round(laya.p_fraud, 4),
            "gate": laya.gate,
            "status": "ok" if laya.p_fraud is not None else "unavailable",
        },
        "model_agreement": response.agreement,
        "evidence_indicators": response.evidence,
        "event_features": response.event.features(),
        "input_source": response.input.source,
    }
    if context:
        parameters["input_context"] = context
    if response.input.raw and response.input.source == "text":
        parameters["analyst_prompt"] = response.input.raw[:MAX_RAW_FOR_QWEN]
    return {"case_id": response.case_id, "verdict": response.verdict, "parameters": parameters}


def _attach_report(response: AnalyzeResponse) -> AnalyzeResponse:
    payload = build_report_payload(response)
    started = time.perf_counter()
    error = None
    model = None
    try:
        report, source, model = _report_parts(generate_report(payload))
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        report = mocks.template_report(payload)
        source = "template"
    return response.model_copy(update={
        "report": report,
        "report_source": source,
        "report_model": model if source == "qwen" else "template",
        "report_latency_ms": (time.perf_counter() - started) * 1000,
        "report_payload": payload,
        "report_error": error,
    })


# --- public API ------------------------------------------------------------------------------
def analyze_batch(
    items: Sequence[tuple[Event, InputInfo | None]],
    auto_reports: int | None = None,
    background: bool = False,
) -> list[AnalyzeResponse]:
    if not items:
        return []
    config = mocks.get_runtime_config()
    events = [event for event, _ in items]
    ml_results = score_events(events)
    laya_results = decide_events(events)
    limit = _auto_report_limit() if auto_reports is None else auto_reports

    responses = []
    for (event, info), ml, laya in zip(items, ml_results, laya_results):
        p_laya = laya.p_fraud
        final = _final_score(p_laya, ml.risk_score, config)
        laya_flag = p_laya is not None and p_laya >= config.laya_fraud_threshold
        ml_flag = ml.risk_score >= ML_HIGH
        fraud_detected = laya_flag or ml_flag
        agreement = ("laya_unavailable" if p_laya is None
                     else "agree" if laya_flag == ml_flag else "disagree")
        responses.append(AnalyzeResponse(
            case_id=uuid4().hex,
            event=event,
            ml=ml,
            laya=laya,
            final=final,
            evidence=extract_evidence(event),
            fraud_detected=fraud_detected,
            verdict="flagged" if fraud_detected else "not_flagged",
            agreement=agreement,
            report=None,
            report_source="none",
            report_latency_ms=None,
            input=info or InputInfo(),
            created_at=mocks.utc_now_iso(),
        ))

    # Reports: flagged first (highest score first), then Laya human-review cases.
    wants_report = [
        i for i, r in enumerate(responses)
        if r.fraud_detected or r.laya.gate == "human_review"
    ]
    wants_report.sort(key=lambda i: (not responses[i].fraud_detected, -responses[i].final.score))
    queued: list[AnalyzeResponse] = []
    for count, index in enumerate(wants_report):
        response = responses[index]
        if count < limit and background:
            # Qwen takes ~15-20 s per report on the Kaggle T4: show the template now and let the
            # background worker swap in the Qwen report (the UI polls the case).
            payload = build_report_payload(response)
            responses[index] = response.model_copy(update={
                "report": mocks.template_report(payload),
                "report_source": "pending",
                "report_model": "template",
                "report_latency_ms": None,
                "report_payload": payload,
                "report_error": None,
            })
            queued.append(responses[index])
        elif count < limit:
            responses[index] = _attach_report(response)
        elif response.fraud_detected:
            # over the Qwen budget: instant template now, Qwen on demand from the UI
            payload = build_report_payload(response)
            responses[index] = response.model_copy(update={
                "report": mocks.template_report(payload),
                "report_source": "template",
                "report_model": "template",
                "report_latency_ms": 0.0,
                "report_payload": payload,
                "report_error": "Qwen auto-report budget for this upload used; request one on the case",
            })

    for response in responses:
        save_case(response)
    for response in queued:
        _report_queue().put(response.case_id)
    return responses


# --- background Qwen worker ------------------------------------------------------------------
_queue: queue.Queue[str] | None = None
_queue_lock = threading.Lock()


def _report_queue() -> "queue.Queue[str]":
    global _queue
    with _queue_lock:
        if _queue is None:
            _queue = queue.Queue()
            threading.Thread(target=_report_worker, args=(_queue,), daemon=True).start()
    return _queue


def _report_worker(jobs: "queue.Queue[str]") -> None:
    while True:
        case_id = jobs.get()
        try:
            case = get_case(case_id)
            if case is not None and case.get("report_source") == "pending":
                done = _attach_report(AnalyzeResponse.model_validate(case))
                _store_report(done)
        except Exception:
            pass
        finally:
            jobs.task_done()


def _store_report(done: AnalyzeResponse) -> None:
    """Write only the report fields, so a review made meanwhile is not overwritten."""
    latest = get_case(done.case_id)
    base = AnalyzeResponse.model_validate(latest) if latest is not None else done
    update_case(base.model_copy(update={
        key: getattr(done, key)
        for key in ("report", "report_source", "report_model", "report_latency_ms",
                    "report_payload", "report_error")
    }))


def wait_for_reports() -> None:
    """Block until queued background reports are written (tests, scripts)."""
    if _queue is not None:
        _queue.join()


def analyze(event: Event, info: InputInfo | None = None) -> AnalyzeResponse:
    return analyze_batch([(event, info)], auto_reports=1)[0]


def regenerate_report(case: dict[str, Any]) -> AnalyzeResponse:
    response = _attach_report(AnalyzeResponse.model_validate(case))
    _store_report(response)
    return AnalyzeResponse.model_validate(get_case(response.case_id))


def laya_checkpoint() -> str:
    if mocks.get_runtime_config().mock_models:
        return "mock"
    if _laya_module is None:
        return "unknown"
    return _laya_module.CHECKPOINT or str(_laya_module.checkpoint_path())


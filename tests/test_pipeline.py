from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from agents.schemas import Event, LayaResult, MLResult
from api import db, main, pipeline


def make_event() -> Event:
    return Event(
        event_id="event-1",
        msisdn_hash="hashed-subscriber-id",
        timestamp="2026-11-13T12:00:00Z",
        imei_changed=1,
        request_hour=3,
        location_gap_km=1200,
        failed_auth_count=4,
        support_calls_24h=2,
        days_since_last_swap=5,
        otp_burst_count=6,
    )


def install_pipeline_stubs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    ml_score: float = 30.0,
    p_laya: float | None = 0.95,
) -> list[dict[str, Any]]:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    monkeypatch.setenv("MOCK_MODELS", "0")
    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.5")
    monkeypatch.setenv("LAYA_WEIGHT", "0.7")
    db.init_db()

    monkeypatch.setattr(
        pipeline,
        "score_event",
        lambda event: MLResult(
            risk_score=ml_score,
            tier="low" if ml_score < 40 else "high",
            lgbm_prob=ml_score / 100,
            anomaly=0.1,
            contributions={},
            latency_ms=1.0,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "decide",
        lambda event: LayaResult(p_fraud=p_laya, model="test-model", latency_ms=1.0),
    )
    monkeypatch.setattr(pipeline, "extract_evidence", lambda event: ["test indicator"])
    captured_payloads = []

    def report(payload: dict[str, Any]) -> tuple[str, str]:
        captured_payloads.append(payload)
        return "A test report for the human analyst.", "qwen"

    monkeypatch.setattr(pipeline, "generate_report", report)
    return captured_payloads


def test_fraud_and_normal_report_behavior(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)
    fraud = pipeline.analyze(make_event())
    assert fraud.report is not None
    assert fraud.report_source in {"qwen", "template"}

    normal_dir = tmp_path / "normal"
    normal_dir.mkdir()
    install_pipeline_stubs(monkeypatch, normal_dir, ml_score=20, p_laya=0.05)
    normal = pipeline.analyze(make_event())
    assert normal.report is None
    assert normal.report_source == "none"
    assert normal.report_latency_ms is None


def test_qwen_failure_uses_template(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)

    def fail(payload: dict[str, Any]) -> None:
        raise TimeoutError("report timed out")

    monkeypatch.setattr(pipeline, "generate_report", fail)
    result = pipeline.analyze(make_event())

    assert result.report is not None
    assert result.report_source == "template"
    assert result.report_latency_ms is not None


def test_laya_unavailable_uses_ml_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=42, p_laya=None)
    result = pipeline.analyze(make_event())

    assert result.agreement == "laya_unavailable"
    assert result.final.score == result.ml.risk_score
    assert result.final.laya_weight_used == 0.0


def test_qwen_payload_uses_final_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payloads = install_pipeline_stubs(monkeypatch, tmp_path, ml_score=30, p_laya=0.9)
    result = pipeline.analyze(make_event())

    assert payloads[0]["risk_score"] == result.final.score


def test_env_threshold_and_weight_affect_final_score(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=30, p_laya=0.4)
    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.3")
    high_threshold = pipeline.analyze(make_event())
    assert high_threshold.fraud_detected

    monkeypatch.setenv("LAYA_FRAUD_THRESHOLD", "0.7")
    low_threshold = pipeline.analyze(make_event())
    assert not low_threshold.fraud_detected
    assert high_threshold.final.score != low_threshold.final.score

    monkeypatch.setenv("LAYA_WEIGHT", "0.2")
    lower_weight = pipeline.analyze(make_event())
    monkeypatch.setenv("LAYA_WEIGHT", "0.9")
    higher_weight = pipeline.analyze(make_event())
    assert lower_weight.final.score != higher_weight.final.score


def test_every_final_score_is_bounded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path, ml_score=100, p_laya=1)
    response = pipeline.analyze(make_event())
    assert 0 <= response.final.score <= 100


def test_mock_mode_runs_without_live_model_dependencies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("DB_PATH", str(tmp_path / "mock.db"))
    monkeypatch.setenv("MOCK_MODELS", "1")
    monkeypatch.delenv("LAYA_FRAUD_THRESHOLD", raising=False)
    monkeypatch.delenv("LAYA_WEIGHT", raising=False)

    with TestClient(main.app) as client:
        fraud = client.post("/analyze", json=make_event().model_dump())
        assert fraud.status_code == 200
        assert fraud.json()["ml"]["risk_score"] == 90
        assert fraud.json()["evidence"] == []
        assert fraud.json()["report_source"] == "template"

        normal_event = make_event().model_copy(
            update={
                "imei_changed": 0,
                "location_gap_km": 10,
                "otp_burst_count": 1,
            }
        )
        normal = client.post("/analyze", json=normal_event.model_dump())
        assert normal.status_code == 200
        assert normal.json()["ml"]["risk_score"] == 20
        assert normal.json()["report"] is None


def test_api_analyze_list_review_and_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    install_pipeline_stubs(monkeypatch, tmp_path)
    with TestClient(main.app) as client:
        analyzed = client.post("/analyze", json=make_event().model_dump())
        assert analyzed.status_code == 200
        case = analyzed.json()
        case_id = case["case_id"]

        monkeypatch.setattr(
            pipeline,
            "score_event",
            lambda event: MLResult(
                risk_score=90,
                tier="high",
                lgbm_prob=0.9,
                anomaly=0.9,
                contributions={},
                latency_ms=1.0,
            ),
        )
        higher = client.post("/analyze", json=make_event().model_dump())
        assert higher.status_code == 200
        higher_case_id = higher.json()["case_id"]

        cases = client.get("/cases")
        assert cases.status_code == 200
        assert [item["case_id"] for item in cases.json()] == [
            higher_case_id,
            case_id,
        ]

        reviewed = client.post(
            f"/cases/{case_id}/review", json={"status": "confirmed"}
        )
        assert reviewed.status_code == 200
        assert reviewed.json()["status"] == "confirmed"

        bad_review = client.post(
            f"/cases/{case_id}/review", json={"status": "invalid"}
        )
        assert bad_review.status_code == 422
        assert client.get("/cases/unknown-case").status_code == 404

    with db._connect() as connection:
        actions = [
            row["action"]
            for row in connection.execute(
                "SELECT action FROM audit_log WHERE case_id = ?", (case_id,)
            )
        ]
    assert actions == ["analyze", "review"]

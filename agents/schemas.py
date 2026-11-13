from typing import Literal, Optional

from pydantic import BaseModel


class Event(BaseModel):
    event_id: str
    msisdn_hash: str
    timestamp: str
    imei_changed: int
    request_hour: int
    location_gap_km: float
    failed_auth_count: int
    support_calls_24h: int
    days_since_last_swap: int
    otp_burst_count: int


class MLResult(BaseModel):
    risk_score: float
    tier: Literal["low", "medium", "high"]
    lgbm_prob: float
    anomaly: float
    contributions: dict[str, float]
    latency_ms: float


class LayaResult(BaseModel):
    p_fraud: Optional[float]
    model: str
    latency_ms: float


class FinalScore(BaseModel):
    score: float  # 0-100, rounded to 1 decimal, shown as "risk score"
    tier: Literal["low", "medium", "high"]
    laya_z: Optional[float]
    ml_z: float
    laya_weight_used: float


class AnalyzeResponse(BaseModel):
    case_id: str
    event: Event
    ml: MLResult
    laya: LayaResult
    final: FinalScore
    evidence: list[str]
    fraud_detected: bool
    agreement: Literal["agree", "disagree", "laya_unavailable"]
    report: Optional[str]
    report_source: Literal["qwen", "template", "none"]
    report_latency_ms: Optional[float]
    status: Literal["pending", "confirmed", "false_alarm"] = "pending"
    created_at: str

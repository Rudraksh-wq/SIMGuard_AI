from typing import Literal, Optional
from pydantic import BaseModel

class Event(BaseModel):
    event_id: str
    msisdn_hash: str
    timestamp: str                 # ISO 8601
    imei_changed: int              # 0/1
    request_hour: int              # 0-23
    location_gap_km: float
    failed_auth_count: int
    support_calls_24h: int
    days_since_last_swap: int
    otp_burst_count: int

class MLResult(BaseModel):
    risk_score: float              # 0-100
    tier: Literal["low", "medium", "high"]
    lgbm_prob: float
    anomaly: float                 # 0-1
    contributions: dict[str, float]   # feature -> LightGBM contribution (pred_contrib)
    latency_ms: float

class LayaResult(BaseModel):
    p_fraud: Optional[float]       # None if Laya failed
    model: str                     # which checkpoint actually answered
    latency_ms: float

class AnalyzeResponse(BaseModel):
    case_id: str
    event: Event
    ml: MLResult
    laya: LayaResult
    evidence: list[str]
    fraud_detected: bool
    agreement: Literal["agree", "disagree", "laya_unavailable"]
    report: Optional[str]
    report_source: Literal["qwen", "template", "none"]
    report_latency_ms: Optional[float]
    status: Literal["pending", "confirmed", "false_alarm"] = "pending"
    created_at: str
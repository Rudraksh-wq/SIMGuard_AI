"""Shared data contracts. Event features match the training dataset, LightGBM, Isolation
Forest, the evidence rules and the fine-tuned Laya model (one schema everywhere)."""

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field


class Event(BaseModel):
    # identifiers (never used as model features)
    event_id: str = ""
    msisdn_hash: str = ""
    timestamp: str = ""
    # the 11 model features (see agents.features_text.FEATURE_COLUMNS)
    sim_swap_time_gap_minutes: float = Field(ge=0)
    device_change_flag: int = Field(ge=0, le=1)
    sim_type_change_flag: int = Field(ge=0, le=1)
    imsi_change_flag: int = Field(ge=0, le=1)
    iccid_change_flag: int = Field(ge=0, le=1)
    otp_and_sim_change_geo_hash_length: float = Field(ge=0, le=1)
    recent_sim_activation_days: float = Field(ge=0)
    num_sim_changes_last_30d: float = Field(ge=0)
    previous_sim_holder_tenure_days: float = Field(ge=0)
    account_age_days: float = Field(ge=0)
    ip_change_flag: int = Field(ge=0, le=1)

    def features(self) -> dict[str, float | int]:
        from agents.features_text import FEATURE_COLUMNS

        return {column: getattr(self, column) for column in FEATURE_COLUMNS}


class MLResult(BaseModel):
    risk_score: float  # 0-100
    tier: Literal["low", "medium", "high"]
    lgbm_prob: float
    anomaly: float  # 0-1, scaled Isolation Forest score
    contributions: dict[str, float]  # feature -> LightGBM pred_contrib (log-odds)
    latency_ms: float


class LayaResult(BaseModel):
    p_fraud: Optional[float]  # None if Laya failed
    model: str  # which checkpoint actually answered
    latency_ms: float
    gate: Optional[Literal["block", "allow", "human_review"]] = None
    error: Optional[str] = None


class FinalScore(BaseModel):
    score: float  # 0-100, rounded to 1 decimal, shown as "risk score"
    tier: Literal["low", "medium", "high"]
    laya_z: Optional[float]
    ml_z: float
    laya_weight_used: float


class InputInfo(BaseModel):
    """Where the event came from and what the parser did with it."""

    source: Literal["api", "json", "csv", "text"] = "api"
    raw: Optional[str] = None  # the original row / prompt (truncated)
    extra: dict[str, Any] = {}  # non-feature columns kept as context (e.g. labels, ids)
    notes: list[str] = []  # parser notes (aliases used, conversions)


class AnalyzeResponse(BaseModel):
    case_id: str
    event: Event
    ml: MLResult
    laya: LayaResult
    final: FinalScore
    evidence: list[str]
    fraud_detected: bool
    verdict: Literal["flagged", "not_flagged", "none"] = "none"
    agreement: Literal["agree", "disagree", "laya_unavailable"]
    report: Optional[str]
    report_source: Literal["qwen", "template", "pending", "none"]
    report_latency_ms: Optional[float]
    report_model: Optional[str] = None
    report_payload: Optional[dict[str, Any]] = None  # exactly what was sent to Qwen
    report_error: Optional[str] = None
    input: InputInfo = InputInfo()
    status: Literal["pending", "confirmed", "false_alarm"] = "pending"
    created_at: str

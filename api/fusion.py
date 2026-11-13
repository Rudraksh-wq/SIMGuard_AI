import math


EPS = 0.01
ML_HIGH = 70.0


def logit(x: float) -> float:
    x = min(max(x, EPS), 1 - EPS)
    return math.log(x / (1 - x))


def compute_final(
    p_laya: float | None,
    ml_risk: float,
    t: float,
    w_laya: float,
) -> tuple[float, float | None, float, float]:
    """Return (score 0-100, z_laya or None, z_ml, w_used)."""
    if not 0 < t < 1:
        raise ValueError("LAYA_FRAUD_THRESHOLD must be strictly between 0 and 1")
    if not 0 <= w_laya <= 1:
        raise ValueError("LAYA_WEIGHT must be between 0 and 1")

    z_ml = logit(ml_risk / 100) - logit(ML_HIGH / 100)
    if p_laya is None:
        return ml_risk, None, z_ml, 0.0

    z_laya = logit(p_laya) - logit(t)
    z = w_laya * z_laya + (1 - w_laya) * z_ml
    score = 100 / (1 + math.exp(-(z + logit(ML_HIGH / 100))))
    return score, z_laya, z_ml, w_laya


def tier_for(score: float) -> str:
    if score < 40:
        return "low"
    if score < 70:
        return "medium"
    return "high"

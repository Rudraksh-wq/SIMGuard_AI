import pytest

from api.fusion import compute_final, tier_for
from api.mocks import get_runtime_config


@pytest.mark.parametrize(
    ("p_laya", "ml_risk", "expected", "tier"),
    [
        (0.95, 90, 96.5, "high"),
        (0.90, 30, 86.7, "high"),
        (0.55, 5, 46.3, "medium"),
        (0.10, 90, 42.9, "medium"),
        (0.02, 10, 5.8, "low"),
    ],
)
def test_weighted_log_odds_examples(
    p_laya: float, ml_risk: float, expected: float, tier: str
) -> None:
    score, _, _, _ = compute_final(p_laya, ml_risk, 0.5, 0.7)
    rounded = round(score, 1)
    assert rounded == pytest.approx(expected, abs=0.1)
    assert tier_for(rounded) == tier
    assert 0 <= rounded <= 100


@pytest.mark.parametrize("ml_risk", [0, 30, 100])
def test_missing_laya_preserves_ml_score(ml_risk: float) -> None:
    score, z_laya, _, weight = compute_final(None, ml_risk, 0.5, 0.7)

    assert score == ml_risk
    assert z_laya is None
    assert weight == 0.0


def test_baseline_point_and_zero_laya_weight() -> None:
    score, _, _, _ = compute_final(0.5, 70, 0.5, 0.7)
    assert round(score, 1) == 70.0

    for ml_risk in [5, 30, 70, 90]:
        score, _, _, _ = compute_final(0.9, ml_risk, 0.5, 0.0)
        assert score == pytest.approx(ml_risk)


@pytest.mark.parametrize(("threshold", "expected"), [(0.3, 65.7), (0.7, 36.9)])
def test_threshold_changes_fused_score(threshold: float, expected: float) -> None:
    score, _, _, _ = compute_final(0.4, 30, threshold, 0.7)
    assert round(score, 1) == pytest.approx(expected, abs=0.1)


@pytest.mark.parametrize(("weight", "expected"), [(0.5, 75.0), (0.9, 93.4)])
def test_weight_changes_fused_score(weight: float, expected: float) -> None:
    score, _, _, _ = compute_final(0.9, 30, 0.5, weight)
    assert round(score, 1) == pytest.approx(expected, abs=0.1)


def test_runtime_config_rejects_invalid_weight(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAYA_WEIGHT", "1.5")
    with pytest.raises(ValueError, match="LAYA_WEIGHT must be between 0 and 1"):
        get_runtime_config()


@pytest.mark.parametrize("p_laya", [None, 0.0, 0.02, 0.5, 0.95, 1.0])
@pytest.mark.parametrize("ml_risk", [0, 5, 30, 70, 90, 100])
def test_scores_stay_in_range(p_laya: float | None, ml_risk: float) -> None:
    score, _, _, _ = compute_final(p_laya, ml_risk, 0.5, 0.7)
    assert 0 <= score <= 100

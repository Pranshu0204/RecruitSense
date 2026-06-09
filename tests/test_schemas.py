"""Schema, helper, and tier-mapping tests for ``backend.core.schemas``."""

import pytest
from pydantic import ValidationError

from backend.core.schemas import (
    DIMENSION_NAMES,
    DIMENSION_WEIGHTS,
    DimensionScore,
    JDInput,
    RecommendedAction,
    ResumeInput,
    ScoreOutput,
    Tier,
    action_from_tier,
    composite_from_dimensions,
    tier_from_composite,
)


def test_dimension_weights_sum_to_one() -> None:
    """Composite math is only valid if weights sum to 1.0."""
    assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (95.0, Tier.A),
        (85.0, Tier.A),
        (84.99, Tier.B),
        (70.0, Tier.B),
        (69.99, Tier.C),
        (55.0, Tier.C),
        (54.99, Tier.D),
        (0.0, Tier.D),
    ],
)
def test_tier_from_composite_thresholds(score: float, expected: Tier) -> None:
    """Tier boundaries: A>=85, B>=70, C>=55, else D."""
    assert tier_from_composite(score) == expected


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        (Tier.A, RecommendedAction.STRONG_HIRE),
        (Tier.B, RecommendedAction.INTERVIEW),
        (Tier.C, RecommendedAction.MAYBE),
        (Tier.D, RecommendedAction.REJECT),
    ],
)
def test_action_from_tier(tier: Tier, expected: RecommendedAction) -> None:
    assert action_from_tier(tier) == expected


def _full_dim_scores(value: float = 7.0) -> dict[str, DimensionScore]:
    return {name: DimensionScore(score=value, rationale="ok") for name in DIMENSION_NAMES}


def test_composite_uniform_seven_yields_seventy() -> None:
    """All-7s should give 70.0 (since weights sum to 1 and we ×10 the result)."""
    assert composite_from_dimensions(_full_dim_scores(7.0)) == 70.0


def test_composite_zero_yields_zero() -> None:
    assert composite_from_dimensions(_full_dim_scores(0.0)) == 0.0


def test_score_output_rejects_missing_dimension() -> None:
    """``ScoreOutput`` validator must reject incomplete dimension dicts."""
    bad = {n: DimensionScore(score=5.0, rationale="x") for n in DIMENSION_NAMES[:-1]}
    with pytest.raises(ValidationError):
        ScoreOutput(
            candidate_name="X",
            composite_score=50.0,
            tier=Tier.D,
            dimension_scores=bad,
            recommended_action=RecommendedAction.REJECT,
            confidence=0.8,
        )


def test_score_output_rejects_unexpected_dimension() -> None:
    """Extra dimension keys must also be rejected."""
    bad = _full_dim_scores()
    bad["bogus_dim"] = DimensionScore(score=5.0, rationale="x")
    with pytest.raises(ValidationError):
        ScoreOutput(
            candidate_name="X",
            composite_score=50.0,
            tier=Tier.D,
            dimension_scores=bad,
            recommended_action=RecommendedAction.REJECT,
            confidence=0.8,
        )


def test_dimension_score_clamps_to_zero_ten() -> None:
    """``score`` is bounded to [0, 10] by Pydantic."""
    with pytest.raises(ValidationError):
        DimensionScore(score=10.5, rationale="x")
    with pytest.raises(ValidationError):
        DimensionScore(score=-0.1, rationale="x")


def test_jd_requires_minimum_description_length() -> None:
    with pytest.raises(ValidationError):
        JDInput(job_title="Eng", description="short")


def test_resume_input_requires_text_or_path() -> None:
    """``ResumeInput`` is invalid with neither ``raw_text`` nor ``file_path``."""
    with pytest.raises(ValidationError):
        ResumeInput(candidate_name="X")
    # Either alone is fine
    ResumeInput(candidate_name="X", raw_text="hello world")
    ResumeInput(candidate_name="X", file_path="/tmp/cv.pdf")

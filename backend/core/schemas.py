"""Pydantic v2 schemas and scoring helpers for RecruitSense.

Defines all I/O contracts (JDInput, ParsedResume, ScoreOutput, BatchResult),
the five scoring dimensions with their fixed weights, and the composite/tier/action
helper functions used across the pipeline.
"""

from datetime import UTC, datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --- Scoring dimensions (single source of truth) ------------------------------

SCORING_DIMENSIONS: tuple[tuple[str, float], ...] = (
    ("skills_match", 0.35),
    ("experience_relevance", 0.30),
    ("education_and_certs", 0.15),
    ("project_impact", 0.10),
    ("communication_and_polish", 0.10),
)
DIMENSION_NAMES: tuple[str, ...] = tuple(name for name, _ in SCORING_DIMENSIONS)
DIMENSION_WEIGHTS: dict[str, float] = dict(SCORING_DIMENSIONS)
assert abs(sum(DIMENSION_WEIGHTS.values()) - 1.0) < 1e-6, "Scoring weights must sum to 1.0"


# --- Enums --------------------------------------------------------------------
class Tier(str, Enum):
    """Candidate tier derived from composite score."""

    A = "A"
    B = "B"
    C = "C"
    D = "D"


class EducationLevel(str, Enum):
    """Minimum education level required by a JD or held by a candidate."""

    HIGH_SCHOOL = "high_school"
    ASSOCIATE = "associate"
    BACHELOR = "bachelor"
    MASTER = "master"
    PHD = "phd"
    NONE = "none"


class RecommendedAction(str, Enum):
    """Action the recruiter should take, derived from tier."""

    STRONG_HIRE = "strong_hire"
    INTERVIEW = "interview"
    MAYBE = "maybe"
    REJECT = "reject"


# --- Input schemas ------------------------------------------------------------
class JDInput(BaseModel):
    """Job description input from the recruiter."""

    job_title: str = Field(..., min_length=1, max_length=200)
    company: str = Field(default="", max_length=200)
    description: str = Field(..., min_length=10)
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience_years: float = Field(default=0.0, ge=0.0, le=50.0)
    education_level: EducationLevel = Field(default=EducationLevel.NONE)


class ResumeInput(BaseModel):
    """Single resume input — exactly one of ``raw_text`` or ``file_path`` required."""

    candidate_name: str = Field(..., min_length=1, max_length=200)
    raw_text: str | None = None
    file_path: str | None = None

    @model_validator(mode="after")
    def _require_text_or_path(self) -> "ResumeInput":
        if not self.raw_text and not self.file_path:
            raise ValueError("ResumeInput requires either `raw_text` or `file_path`")
        return self


# --- Internal parsed-resume model --------------------------------------------
class ParsedResume(BaseModel):
    """Structured fields extracted from raw resume text by the parser agent."""

    candidate_name: str
    skills: list[str] = Field(default_factory=list)
    experience: list[str] = Field(default_factory=list)
    education: list[str] = Field(default_factory=list)
    certifications: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)
    total_years_experience: float = Field(default=0.0, ge=0.0, le=70.0)


# --- Output schemas -----------------------------------------------------------
class DimensionScore(BaseModel):
    """Score and rationale for a single scoring dimension."""

    score: float = Field(..., ge=0.0, le=10.0)
    rationale: str = Field(..., min_length=1, max_length=2000)


class ScoreOutput(BaseModel):
    """Full scoring result for a single candidate."""

    # `model_used` would otherwise collide with Pydantic's protected `model_` namespace.
    model_config = ConfigDict(protected_namespaces=())

    candidate_name: str
    composite_score: float = Field(..., ge=0.0, le=100.0)
    tier: Tier
    dimension_scores: dict[str, DimensionScore]
    top_strengths: list[str] = Field(default_factory=list, max_length=10)
    key_gaps: list[str] = Field(default_factory=list, max_length=10)
    bias_flags: list[str] = Field(default_factory=list)
    recommended_action: RecommendedAction
    rag_context_used: str = Field(default="")
    confidence: float = Field(..., ge=0.0, le=1.0)
    model_used: str = Field(default="")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("dimension_scores")
    @classmethod
    def _validate_dimension_keys(cls, v: dict[str, DimensionScore]) -> dict[str, DimensionScore]:
        missing = set(DIMENSION_NAMES) - set(v.keys())
        if missing:
            raise ValueError(f"Missing dimension scores: {sorted(missing)}")
        unexpected = set(v.keys()) - set(DIMENSION_NAMES)
        if unexpected:
            raise ValueError(f"Unexpected dimensions: {sorted(unexpected)}")
        return v


class BatchResult(BaseModel):
    """Aggregate result for a batch screening request."""

    job_title: str
    total_resumes: int = Field(..., ge=0)
    ranked_candidates: list[ScoreOutput] = Field(default_factory=list)
    tier_distribution: dict[str, int] = Field(default_factory=dict)
    shortlisted_count: int = Field(..., ge=0)
    processing_time_seconds: float = Field(..., ge=0.0)


# --- Helpers ------------------------------------------------------------------
def composite_from_dimensions(
    dim_scores: dict[str, DimensionScore],
    weight_overrides: dict[str, float] | None = None,
) -> float:
    """Compute the 0-100 composite from per-dimension 0-10 scores.

    Pass ``weight_overrides`` (must cover all five dimensions and sum to 1.0)
    to recompute with custom weights without re-invoking the LLM.
    """
    weights = weight_overrides if weight_overrides is not None else DIMENSION_WEIGHTS
    weighted = sum(dim_scores[name].score * weights[name] for name in DIMENSION_NAMES)
    return round(weighted * 10.0, 2)


def tier_from_composite(score: float) -> Tier:
    """Map a 0-100 composite score to a tier (A>=85, B>=70, C>=55, else D)."""
    if score >= 85:
        return Tier.A
    if score >= 70:
        return Tier.B
    if score >= 55:
        return Tier.C
    return Tier.D


# --- Session schemas (stateful LangGraph turns) ------------------------------


class SessionScreenResponse(BaseModel):
    """Returned by Turn 1 (/session/screen) — score plus the session_id needed for Turns 2 & 3."""

    session_id: str
    score: ScoreOutput


class ReweightRequest(BaseModel):
    """Body for Turn 2 (/session/{id}/reweight) — full set of dimension weights."""

    weight_overrides: dict[str, float]

    @model_validator(mode="after")
    def _validate_weights(self) -> "ReweightRequest":
        invalid = set(self.weight_overrides) - set(DIMENSION_NAMES)
        if invalid:
            raise ValueError(f"Unknown dimensions: {sorted(invalid)}")
        missing = set(DIMENSION_NAMES) - set(self.weight_overrides)
        if missing:
            raise ValueError(f"Missing dimensions: {sorted(missing)}")
        total = sum(self.weight_overrides.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total:.3f}")
        return self


class DimensionComparison(BaseModel):
    """Per-dimension winner between two candidates."""

    candidate_a_score: float
    candidate_b_score: float
    winner: str
    delta: float


class CompareRequest(BaseModel):
    """Body for Turn 3 (/session/compare) — two session IDs to compare."""

    session_id_a: str
    session_id_b: str


class CompareResponse(BaseModel):
    """Side-by-side comparison of two saved screening sessions."""

    candidate_a: ScoreOutput
    candidate_b: ScoreOutput
    overall_winner: str
    score_delta: float
    dimension_comparison: dict[str, DimensionComparison]


def action_from_tier(tier: Tier) -> RecommendedAction:
    """Map a tier to the recommended recruiter action."""
    return {
        Tier.A: RecommendedAction.STRONG_HIRE,
        Tier.B: RecommendedAction.INTERVIEW,
        Tier.C: RecommendedAction.MAYBE,
        Tier.D: RecommendedAction.REJECT,
    }[tier]

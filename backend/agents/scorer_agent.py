"""Scorer agent — scores a parsed resume against a JD using RAG context.

Pipeline:
1. Format the scorer prompt with JD + parsed resume + RAG context.
2. LLM call (cached by sha256 of all inputs + ``PROMPT_VERSION``).
3. Strip the ``<thinking>...</thinking>`` chain-of-thought block (kept private).
4. Parse the trailing JSON, validate dimensions, compute composite via
   :func:`composite_from_dimensions`, derive tier and recommended action.
5. Attach pre-computed bias flags to the output (these are NOT used in the LLM
   prompt — bias signals must never influence the score).
6. On any failure, return a fallback ``ScoreOutput`` with ``confidence=0.0``
   so a single broken candidate never breaks a batch.
"""

from __future__ import annotations

import json
import re
from typing import Any

from backend.agents import get_llm
from backend.core.config import get_settings
from backend.core.prompts import SCORER_PROMPT
from backend.core.schemas import (
    DIMENSION_NAMES,
    DimensionScore,
    JDInput,
    ParsedResume,
    ScoreOutput,
    action_from_tier,
    composite_from_dimensions,
    tier_from_composite,
)
from backend.utils.logger import get_logger
from backend.utils.redis_cache import cache_llm

logger = get_logger(__name__)


# --- Output cleaning ----------------------------------------------------------

_THINKING_BLOCK = re.compile(r"<thinking>.*?</thinking>", re.DOTALL | re.IGNORECASE)
_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def _strip_thinking_block(text: str) -> str:
    return _THINKING_BLOCK.sub("", text).strip()


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text)
        text = _FENCE_CLOSE.sub("", text)
    return text.strip()


def _extract_first_json_object(text: str) -> str:
    """Return the substring spanning the first balanced ``{...}`` block, or the original text."""
    start = text.find("{")
    if start < 0:
        return text
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return text[start:]


# --- LLM call -----------------------------------------------------------------


@cache_llm(namespace="scorer")
async def _scorer_llm_call(
    jd_dict: dict[str, Any],
    parsed_resume_dict: dict[str, Any],
    rag_context: str,
) -> dict[str, Any]:
    """Cached scorer LLM call. Returns the parsed JSON object (raises on failure)."""
    llm = get_llm()
    prompt = SCORER_PROMPT.format_messages(
        job_title=jd_dict["job_title"],
        company=jd_dict.get("company", ""),
        required_skills=", ".join(jd_dict.get("required_skills", []) or []),
        preferred_skills=", ".join(jd_dict.get("preferred_skills", []) or []),
        min_experience_years=jd_dict.get("min_experience_years", 0),
        education_level=jd_dict.get("education_level", "none"),
        jd_description=jd_dict["description"],
        parsed_resume_json=json.dumps(parsed_resume_dict, indent=2, default=str),
        rag_context=rag_context or "(no additional context retrieved)",
    )
    response = await llm.ainvoke(prompt)
    content = _strip_thinking_block(str(response.content))
    content = _strip_code_fences(content)
    content = _extract_first_json_object(content)
    return json.loads(content)


# --- Fallback construction ----------------------------------------------------


def _make_fallback_score(
    candidate_name: str, model_used: str, error: str, bias_flags: list[str] | None = None
) -> ScoreOutput:
    """Build a zero-confidence ``ScoreOutput`` when scoring fails."""
    fallback_dims = {
        name: DimensionScore(score=0.0, rationale=f"Scoring failed: {error[:200]}")
        for name in DIMENSION_NAMES
    }
    composite = composite_from_dimensions(fallback_dims)
    tier = tier_from_composite(composite)
    return ScoreOutput(
        candidate_name=candidate_name,
        composite_score=composite,
        tier=tier,
        dimension_scores=fallback_dims,
        top_strengths=[],
        key_gaps=[f"Scoring error: {error[:200]}"],
        bias_flags=bias_flags or [],
        recommended_action=action_from_tier(tier),
        rag_context_used="",
        confidence=0.0,
        model_used=model_used,
    )


# --- Public entry point -------------------------------------------------------


async def score_resume(
    jd: JDInput,
    parsed_resume: ParsedResume,
    rag_context: str = "",
    bias_flags: list[str] | None = None,
) -> ScoreOutput:
    """Score a parsed resume against a JD and return a ``ScoreOutput``.

    ``bias_flags`` are passed through to the output but never sent to the LLM —
    bias signals must never influence the score.
    """
    model_used = get_settings().default_model

    try:
        raw = await _scorer_llm_call(
            jd.model_dump(mode="json"),
            parsed_resume.model_dump(mode="json"),
            rag_context,
        )
    except Exception as exc:
        logger.warning("scorer_llm_failed", reason=str(exc))
        return _make_fallback_score(
            parsed_resume.candidate_name, model_used, str(exc), bias_flags
        )

    try:
        dim_scores = {
            name: DimensionScore(**raw["dimension_scores"][name])
            for name in DIMENSION_NAMES
        }
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("scorer_dimension_parse_failed", reason=str(exc))
        return _make_fallback_score(
            parsed_resume.candidate_name, model_used, str(exc), bias_flags
        )

    composite = composite_from_dimensions(dim_scores)
    tier = tier_from_composite(composite)

    try:
        confidence = float(raw.get("confidence", 0.7))
    except (TypeError, ValueError):
        confidence = 0.5
    confidence = max(0.0, min(1.0, confidence))

    return ScoreOutput(
        candidate_name=parsed_resume.candidate_name,
        composite_score=composite,
        tier=tier,
        dimension_scores=dim_scores,
        top_strengths=list(raw.get("top_strengths") or [])[:10],
        key_gaps=list(raw.get("key_gaps") or [])[:10],
        bias_flags=bias_flags or [],
        recommended_action=action_from_tier(tier),
        rag_context_used=rag_context[:2000] if rag_context else "",
        confidence=confidence,
        model_used=model_used,
    )

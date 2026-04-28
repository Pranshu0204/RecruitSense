"""``POST /batch`` — multi-resume batch screening with bounded concurrency.

Accepts up to :data:`MAX_BATCH_SIZE` resumes (default 50) and screens them in
parallel through ``asyncio.gather``, gated by an :data:`CONCURRENCY` semaphore
(default 10) so the LLM provider isn't overwhelmed. A failed PDF / failed
pipeline produces a zero-confidence ``ScoreOutput`` for that candidate rather
than failing the whole batch.
"""

from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.agents.graph import run_pipeline
from backend.core.schemas import (
    DIMENSION_NAMES,
    BatchResult,
    DimensionScore,
    JDInput,
    ScoreOutput,
    Tier,
    action_from_tier,
    composite_from_dimensions,
    tier_from_composite,
)
from backend.utils.logger import get_logger
from backend.utils.pdf_parser import PDFParseError, extract_text_from_pdf

logger = get_logger(__name__)
router: APIRouter = APIRouter()

MAX_BATCH_SIZE: int = 50
CONCURRENCY: int = 10


def _file_failure_score(filename: str, error: str) -> ScoreOutput:
    """Construct a zero-confidence ``ScoreOutput`` for an unprocessable file."""
    fallback_dims = {
        name: DimensionScore(score=0.0, rationale=f"File error: {error[:200]}")
        for name in DIMENSION_NAMES
    }
    composite = composite_from_dimensions(fallback_dims)
    tier = tier_from_composite(composite)
    return ScoreOutput(
        candidate_name=filename or "unknown.pdf",
        composite_score=composite,
        tier=tier,
        dimension_scores=fallback_dims,
        top_strengths=[],
        key_gaps=[f"Could not process file: {error[:200]}"],
        bias_flags=[],
        recommended_action=action_from_tier(tier),
        rag_context_used="",
        confidence=0.0,
    )


@router.post("/batch", response_model=BatchResult)
async def batch_screen(
    jd_json: str = Form(..., description="JSON-serialized JDInput"),
    resumes: list[UploadFile] = File(
        ..., description=f"Up to {MAX_BATCH_SIZE} resume PDFs"
    ),
) -> BatchResult:
    """Screen many resumes against a single JD; returns a ranked leaderboard."""
    try:
        jd = JDInput.model_validate_json(jd_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    if not resumes:
        raise HTTPException(status_code=400, detail="At least one resume required.")
    if len(resumes) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Maximum {MAX_BATCH_SIZE} resumes per batch.",
        )

    semaphore = asyncio.Semaphore(CONCURRENCY)

    async def _screen_one(rfile: UploadFile) -> ScoreOutput:
        async with semaphore:
            filename = rfile.filename or "unknown.pdf"
            try:
                pdf_bytes = await rfile.read()
                if not pdf_bytes:
                    return _file_failure_score(filename, "empty PDF")
                if not filename.lower().endswith(".pdf"):
                    return _file_failure_score(filename, "not a PDF")
                text = extract_text_from_pdf(pdf_bytes)
                return await run_pipeline(jd, text)
            except PDFParseError as exc:
                return _file_failure_score(filename, str(exc))
            except Exception as exc:  # noqa: BLE001 — never let one resume break the batch
                logger.warning(
                    "batch_resume_failed", filename=filename, reason=str(exc)
                )
                return _file_failure_score(filename, str(exc))

    start = time.perf_counter()
    results: list[ScoreOutput] = await asyncio.gather(
        *[_screen_one(r) for r in resumes]
    )
    elapsed = time.perf_counter() - start

    results.sort(key=lambda r: r.composite_score, reverse=True)

    tier_distribution: dict[str, int] = {t.value: 0 for t in Tier}
    for r in results:
        tier_distribution[r.tier.value] += 1

    shortlisted = sum(1 for r in results if r.tier in {Tier.A, Tier.B})

    logger.info(
        "batch_complete",
        total=len(resumes),
        shortlisted=shortlisted,
        duration_s=round(elapsed, 2),
        tier_distribution=tier_distribution,
    )

    return BatchResult(
        job_title=jd.job_title,
        total_resumes=len(resumes),
        ranked_candidates=results,
        tier_distribution=tier_distribution,
        shortlisted_count=shortlisted,
        processing_time_seconds=round(elapsed, 2),
    )

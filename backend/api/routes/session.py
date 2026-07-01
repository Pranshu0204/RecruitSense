"""Session-based screening endpoints — stateful LangGraph across three turns.

Turn 1  POST /session/screen              Full pipeline. State saved under session_id.
Turn 2  POST /session/{id}/reweight       Recompute composite with new weights.
                                          No LLM call — reads saved dimension scores.
Turn 3  POST /session/compare             Side-by-side comparison of two saved sessions.
                                          No LLM call — reads both saved states.

The three turns demonstrate LangGraph's MemorySaver checkpointing: state written
during Turn 1 is retrieved in Turns 2 and 3 purely from in-memory persistence,
so the LLM is only ever called once per candidate.
"""

from __future__ import annotations

import uuid

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.agents.graph import build_session_graph
from backend.core.schemas import (
    DIMENSION_NAMES,
    CompareRequest,
    CompareResponse,
    DimensionComparison,
    JDInput,
    ReweightRequest,
    ScoreOutput,
    SessionScreenResponse,
    action_from_tier,
    composite_from_dimensions,
    tier_from_composite,
)
from backend.utils.logger import get_logger
from backend.utils.pdf_parser import PDFParseError, extract_text_from_pdf

logger = get_logger(__name__)
router: APIRouter = APIRouter(prefix="/session", tags=["session"])


# --- Turn 1 -------------------------------------------------------------------


@router.post("/screen", response_model=SessionScreenResponse)
async def session_screen(
    jd_json: str = Form(..., description="JSON-serialized JDInput"),
    resume: UploadFile = File(..., description="Candidate resume PDF"),
    model: str = Form(default="", description="OpenRouter model slug"),
) -> SessionScreenResponse:
    """Turn 1 — run the full pipeline and persist state under a new session_id.

    Returns the score plus a ``session_id`` that the recruiter passes to
    Turn 2 (reweight) or Turn 3 (compare) without re-uploading the resume.
    """
    try:
        jd = JDInput.model_validate_json(jd_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    filename = resume.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Resume must be a PDF.")

    pdf_bytes = await resume.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF upload.")

    try:
        resume_text = extract_text_from_pdf(pdf_bytes)
    except PDFParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    session_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": session_id}}

    result = await build_session_graph().ainvoke(
        {"jd": jd, "resume_text": resume_text},
        config=config,
    )

    logger.info(
        "session_screen_complete",
        session_id=session_id,
        candidate=result["score"].candidate_name,
        tier=result["score"].tier.value,
    )
    return SessionScreenResponse(session_id=session_id, score=result["score"])


# --- Turn 2 -------------------------------------------------------------------


@router.post("/{session_id}/reweight", response_model=ScoreOutput)
async def session_reweight(session_id: str, body: ReweightRequest) -> ScoreOutput:
    """Turn 2 — recompute the composite score with new dimension weights.

    Loads saved dimension scores from the checkpointer and recomputes the
    composite in Python — no LLM call, no resume re-upload needed.
    This is the key demonstration of LangGraph stateful persistence: the
    dimension scores written during Turn 1 are retrieved directly from memory.
    """
    snapshot = await build_session_graph().aget_state({"configurable": {"thread_id": session_id}})
    if not snapshot.values.get("score"):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found. Run Turn 1 first.",
        )

    saved_score: ScoreOutput = snapshot.values["score"]
    new_composite = composite_from_dimensions(saved_score.dimension_scores, body.weight_overrides)
    new_tier = tier_from_composite(new_composite)

    logger.info(
        "session_reweight_complete",
        session_id=session_id,
        old_composite=saved_score.composite_score,
        new_composite=new_composite,
        new_tier=new_tier.value,
    )

    return ScoreOutput(
        **{
            **saved_score.model_dump(exclude={"composite_score", "tier", "recommended_action"}),
            "composite_score": new_composite,
            "tier": new_tier,
            "recommended_action": action_from_tier(new_tier),
        }
    )


# --- Turn 3 -------------------------------------------------------------------


@router.post("/compare", response_model=CompareResponse)
async def session_compare(body: CompareRequest) -> CompareResponse:
    """Turn 3 — load two saved sessions and return a side-by-side comparison.

    No LLM call. Both states are read from the MemorySaver checkpointer.
    Returns per-dimension winners and an overall winner by composite score.
    """
    graph = build_session_graph()
    snap_a = await graph.aget_state({"configurable": {"thread_id": body.session_id_a}})
    snap_b = await graph.aget_state({"configurable": {"thread_id": body.session_id_b}})

    if not snap_a.values.get("score"):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id_a}' not found.",
        )
    if not snap_b.values.get("score"):
        raise HTTPException(
            status_code=404,
            detail=f"Session '{body.session_id_b}' not found.",
        )

    score_a: ScoreOutput = snap_a.values["score"]
    score_b: ScoreOutput = snap_b.values["score"]

    delta = round(score_a.composite_score - score_b.composite_score, 2)
    if delta > 0:
        overall_winner = score_a.candidate_name
    elif delta < 0:
        overall_winner = score_b.candidate_name
    else:
        overall_winner = "tie"

    dimension_comparison: dict[str, DimensionComparison] = {}
    for dim in DIMENSION_NAMES:
        s_a = score_a.dimension_scores[dim].score
        s_b = score_b.dimension_scores[dim].score
        d = round(s_a - s_b, 2)
        dimension_comparison[dim] = DimensionComparison(
            candidate_a_score=s_a,
            candidate_b_score=s_b,
            winner="A" if d > 0 else ("B" if d < 0 else "tie"),
            delta=d,
        )

    logger.info(
        "session_compare_complete",
        session_id_a=body.session_id_a,
        session_id_b=body.session_id_b,
        overall_winner=overall_winner,
        score_delta=delta,
    )

    return CompareResponse(
        candidate_a=score_a,
        candidate_b=score_b,
        overall_winner=overall_winner,
        score_delta=delta,
        dimension_comparison=dimension_comparison,
    )

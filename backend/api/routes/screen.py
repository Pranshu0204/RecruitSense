"""``POST /screen`` — single-resume screening pipeline.

Accepts a multipart request with:
- ``jd_json`` (Form, str): JSON-serialized :class:`JDInput`.
- ``resume`` (File, PDF): the candidate's resume.

Runs the LangGraph DAG (parser → rag ∥ bias → scorer) and returns a
:class:`ScoreOutput`.
"""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import ValidationError

from backend.agents.graph import run_pipeline
from backend.core.schemas import JDInput, ScoreOutput
from backend.utils.logger import get_logger
from backend.utils.pdf_parser import PDFParseError, extract_text_from_pdf

logger = get_logger(__name__)
router: APIRouter = APIRouter()


@router.post("/screen", response_model=ScoreOutput)
async def screen_resume(
    jd_json: str = Form(..., description="JSON-serialized JDInput"),
    resume: UploadFile = File(..., description="Candidate resume (PDF)"),
) -> ScoreOutput:
    """Screen a single resume PDF against a JD."""
    try:
        jd = JDInput.model_validate_json(jd_json)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=exc.errors()) from exc

    filename = resume.filename or ""
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400, detail="Resume must be a PDF (.pdf extension required)."
        )

    pdf_bytes = await resume.read()
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty PDF upload.")

    try:
        resume_text = extract_text_from_pdf(pdf_bytes)
    except PDFParseError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    logger.info(
        "screen_request",
        filename=filename,
        jd_title=jd.job_title,
        resume_chars=len(resume_text),
    )
    return await run_pipeline(jd, resume_text)

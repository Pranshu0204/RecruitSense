"""RAG evaluation — context relevance and faithfulness metrics.

Measures two properties of the RAG pipeline without external dependencies:

  Context Relevance  — how well retrieved chunks match each sub-query (0–1)
  Faithfulness       — fraction of scorer rationale claims grounded in the full
                       source set the scorer saw: resume + JD + RAG context (0–1)

Both metrics use the same LLM already wired into the project (via OpenRouter)
so no extra API keys or packages are needed.

Usage:
    python -m backend.rag.evaluate_rag
    python -m backend.rag.evaluate_rag --output eval_report.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

from backend.agents import get_llm
from backend.agents.parser_agent import parse_resume
from backend.agents.rag_agent import (
    _fallback_subqueries,
    _generate_subqueries,
    _infer_industry,
    _infer_seniority,
    _retrieve_one,
    retrieve_context,
)
from backend.agents.scorer_agent import score_resume
from backend.core.schemas import JDInput, ParsedResume
from backend.rag.vector_store import get_store
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Test cases — synthetic but realistic JD + resume pairs
# ---------------------------------------------------------------------------

_TEST_CASES: list[dict[str, Any]] = [
    {
        "name": "Senior Backend Engineer — Fintech",
        "jd": {
            "job_title": "Senior Backend Engineer",
            "company": "PayCore (fintech payments startup)",
            "description": (
                "We are building real-time payment infrastructure. "
                "You will design high-throughput APIs, own database schema, "
                "and mentor junior engineers in a fast-moving fintech environment."
            ),
            "required_skills": ["Python", "FastAPI", "PostgreSQL", "Redis"],
            "preferred_skills": ["Kafka", "Docker", "AWS"],
            "min_experience_years": 5.0,
            "education_level": "bachelor",
        },
        "resume_text": (
            "Jane Smith\n"
            "jane@example.com\n\n"
            "EXPERIENCE\n"
            "Senior Software Engineer @ Stripe (2020 - 2024)\n"
            "  - Designed REST APIs handling 50k req/s using Python and FastAPI\n"
            "  - Led migration of monolithic service to microservices on AWS ECS\n"
            "  - Mentored 3 junior engineers, conducted weekly code reviews\n\n"
            "Backend Engineer @ TransferWise (2018 - 2020)\n"
            "  - Built payment reconciliation pipelines using PostgreSQL and Redis\n"
            "  - Reduced p99 latency from 800ms to 120ms via query optimisation\n\n"
            "SKILLS\n"
            "Python, FastAPI, PostgreSQL, Redis, Kafka, Docker, AWS, REST APIs\n\n"
            "EDUCATION\n"
            "B.Sc. Computer Science, University of Edinburgh (2018)\n\n"
            "CERTIFICATIONS\n"
            "AWS Certified Solutions Architect — Associate"
        ),
    },
    {
        "name": "Junior Data Scientist — Healthcare",
        "jd": {
            "job_title": "Junior Data Scientist",
            "company": "MedAnalytics (healthcare data company)",
            "description": (
                "Join our clinical data team to build predictive models on "
                "patient outcome data. You will work with structured EHR data, "
                "train ML models, and present findings to non-technical stakeholders."
            ),
            "required_skills": ["Python", "scikit-learn", "pandas", "SQL"],
            "preferred_skills": ["PyTorch", "Tableau", "HIPAA compliance"],
            "min_experience_years": 1.0,
            "education_level": "bachelor",
        },
        "resume_text": (
            "Alex Johnson\n"
            "alex.johnson@email.com\n\n"
            "EXPERIENCE\n"
            "Data Science Intern @ HealthFirst (2023 - 2024)\n"
            "  - Built patient readmission prediction model using scikit-learn (AUC 0.84)\n"
            "  - Cleaned and merged EHR datasets using pandas and SQL queries\n"
            "  - Presented model results to clinical operations team\n\n"
            "SKILLS\n"
            "Python, pandas, scikit-learn, SQL, NumPy, Matplotlib, Tableau\n\n"
            "EDUCATION\n"
            "B.Sc. Statistics, University of Michigan (2023)\n\n"
            "PROJECTS\n"
            "ICU Length-of-Stay Prediction — gradient boosted trees on MIMIC-III dataset"
        ),
    },
    {
        "name": "Mid-Level ML Engineer — AI/ML Platform",
        "jd": {
            "job_title": "Machine Learning Engineer",
            "company": "VectorAI (ML infrastructure startup)",
            "description": (
                "Build and maintain ML training pipelines, model serving infrastructure, "
                "and feature stores. Work closely with research to productionise models "
                "at scale in a cloud-native ML platform environment."
            ),
            "required_skills": ["Python", "PyTorch", "MLflow", "Kubernetes"],
            "preferred_skills": ["Ray", "Triton", "Rust"],
            "min_experience_years": 3.0,
            "education_level": "master",
        },
        "resume_text": (
            "Priya Patel\n"
            "priya.patel@dev.io\n\n"
            "EXPERIENCE\n"
            "ML Engineer @ Spotify (2021 - 2024)\n"
            "  - Productionised recommendation model serving 400M users via Kubernetes\n"
            "  - Built feature store integrating real-time and batch features using Ray\n"
            "  - Maintained MLflow tracking for 20+ model experiments per week\n\n"
            "Research Engineer Intern @ DeepMind (2020)\n"
            "  - Implemented custom PyTorch training loops for RL experiments\n\n"
            "SKILLS\n"
            "Python, PyTorch, MLflow, Kubernetes, Ray, Docker, ONNX, Triton\n\n"
            "EDUCATION\n"
            "M.Sc. Machine Learning, Carnegie Mellon University (2021)"
        ),
    },
]


# ---------------------------------------------------------------------------
# LLM-judge throttling
# ---------------------------------------------------------------------------
# Free-tier OpenRouter models (e.g. gpt-oss-120b:free) are aggressively
# rate-limited. The judge fires many small calls, so cap concurrency hard and
# let get_llm()'s built-in retry/backoff absorb transient 429s. A call that
# still fails after retries returns a neutral fallback rather than crashing the
# whole evaluation.
_JUDGE_CONCURRENCY = 2
_judge_semaphore = asyncio.Semaphore(_JUDGE_CONCURRENCY)
# Chunks per sub-query to judge for relevance. Kept small to bound LLM calls.
_RELEVANCE_CHUNKS_PER_QUERY = 5


async def _judge_call(prompt: str) -> str | None:
    """Throttled LLM judge call. Returns the response text, or None on failure."""
    async with _judge_semaphore:
        try:
            llm = get_llm(temperature=0.0)
            response = await llm.ainvoke(prompt)
            return str(response.content).strip()
        except Exception as exc:
            logger.warning("judge_call_failed", reason=str(exc))
            return None


# ---------------------------------------------------------------------------
# Metric 1 — Context Relevance
# ---------------------------------------------------------------------------

_RELEVANCE_PROMPT = """\
You are evaluating a RAG retrieval system.

Sub-query: "{query}"

Retrieved chunk:
\"\"\"
{chunk}
\"\"\"

Rate how relevant this chunk is for answering the sub-query.
Reply with ONLY a decimal number between 0.0 (completely irrelevant) and 1.0 (perfectly relevant).
No explanation. Just the number."""


async def _score_chunk_relevance(query: str, chunk_text: str) -> float | None:
    """Ask the LLM to score one chunk's relevance to a sub-query.

    Returns ``None`` if the judge call failed (so it can be excluded from the
    mean rather than dragging it toward an arbitrary value).
    """
    raw = await _judge_call(_RELEVANCE_PROMPT.format(query=query, chunk=chunk_text[:800]))
    if raw is None:
        return None
    match = re.search(r"\d+\.?\d*", raw)
    if not match:
        return 0.5
    return max(0.0, min(1.0, float(match.group())))


async def compute_context_relevance(
    subqueries: list[str],
    chunks_per_query: list[list[Any]],
) -> float:
    """Mean relevance score across all (sub-query, chunk) pairs."""
    tasks = []
    for query, chunks in zip(subqueries, chunks_per_query, strict=True):
        for chunk in chunks[:_RELEVANCE_CHUNKS_PER_QUERY]:
            tasks.append(_score_chunk_relevance(query, chunk.text))
    if not tasks:
        return 0.0
    results = await asyncio.gather(*tasks)
    scores = [s for s in results if s is not None]
    if not scores:
        return 0.0
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Metric 2 — Faithfulness
# ---------------------------------------------------------------------------

_CLAIM_EXTRACTION_PROMPT = """\
Extract every factual claim made in the following text about skills, experience benchmarks, or role requirements.
List each claim on its own line, starting with "- ".
Keep each claim short (one sentence). Do not add claims that are not in the text.

Text:
\"\"\"
{text}
\"\"\""""

_FAITHFULNESS_CHECK_PROMPT = """\
You are verifying whether a claim is grounded in the source material the scorer was given.

The scorer's legitimate sources are the candidate's resume, the job description, and the
retrieved knowledge-base benchmarks. A claim is "supported" if it is stated in, or directly
inferable from, ANY of these sources. It is "unsupported" only if it appears in none of them
(i.e. the scorer invented it).

Source material:
\"\"\"
{context}
\"\"\"

Claim: "{claim}"

Is this claim supported by or directly inferable from the source material above?
Reply with ONLY "yes" or "no"."""


async def _extract_claims(text: str) -> list[str]:
    """Use the LLM to break rationale text into atomic claims."""
    raw = await _judge_call(_CLAIM_EXTRACTION_PROMPT.format(text=text[:1500]))
    if raw is None:
        return []
    lines = raw.splitlines()
    claims = [line.lstrip("- •\t").strip() for line in lines if line.strip().startswith("-")]
    return [c for c in claims if c]


async def _check_claim_supported(claim: str, grounding: str) -> bool | None:
    """Ask the LLM whether a single claim is supported by the grounding material.

    Returns ``None`` if the judge call failed, so it can be excluded rather than
    counted as unsupported.
    """
    raw = await _judge_call(
        _FAITHFULNESS_CHECK_PROMPT.format(context=grounding[:4000], claim=claim)
    )
    if raw is None:
        return None
    return raw.lower().startswith("yes")


def _build_grounding(jd: JDInput, parsed_resume: ParsedResume, rag_context: str) -> str:
    """Assemble the full set of sources the scorer legitimately drew on.

    Faithfulness must be measured against everything the scorer saw — the
    candidate's parsed resume and the JD, not just the RAG benchmarks — otherwise
    legitimate resume-grounded claims are wrongly flagged as hallucinations.
    """
    jd_block = (
        f"Job title: {jd.job_title}\nCompany: {jd.company}\n"
        f"Required skills: {', '.join(jd.required_skills)}\n"
        f"Preferred skills: {', '.join(jd.preferred_skills)}\n"
        f"Minimum experience: {jd.min_experience_years} years\n"
        f"Description: {jd.description}"
    )
    resume_block = (
        f"Candidate: {parsed_resume.candidate_name}\n"
        f"Skills: {', '.join(parsed_resume.skills)}\n"
        f"Experience: {' | '.join(parsed_resume.experience)}\n"
        f"Education: {' | '.join(parsed_resume.education)}\n"
        f"Certifications: {' | '.join(parsed_resume.certifications)}\n"
        f"Projects: {' | '.join(parsed_resume.projects)}\n"
        f"Total years experience: {parsed_resume.total_years_experience}"
    )
    return (
        f"## JOB DESCRIPTION\n{jd_block}\n\n"
        f"## CANDIDATE RESUME\n{resume_block}\n\n"
        f"## KNOWLEDGE-BASE BENCHMARKS\n{rag_context or '(none retrieved)'}"
    )


async def compute_faithfulness(rationales: list[str], grounding: str) -> float:
    """Faithfulness = supported claims / total checked claims across all rationales.

    ``grounding`` should be the full source set (resume + JD + RAG context) the
    scorer was given — see :func:`_build_grounding`.
    """
    if not grounding:
        return 0.0

    combined = " ".join(rationales)
    claims = await _extract_claims(combined)
    if not claims:
        return 1.0

    results = await asyncio.gather(*[_check_claim_supported(c, grounding) for c in claims])
    checked = [r for r in results if r is not None]
    if not checked:
        return 0.0
    supported = sum(1 for ok in checked if ok)
    return supported / len(checked)


# ---------------------------------------------------------------------------
# Per-case evaluation
# ---------------------------------------------------------------------------


@dataclass
class CaseResult:
    name: str
    context_relevance: float
    faithfulness: float
    n_chunks_retrieved: int
    n_claims_checked: int
    subqueries: list[str] = field(default_factory=list)
    tier: str = ""
    composite_score: float = 0.0


async def evaluate_case(case: dict[str, Any]) -> CaseResult:
    jd = JDInput(**case["jd"])
    resume_text: str = case["resume_text"]
    name: str = case["name"]

    logger.info("eval_case_start", case=name)

    # --- RAG retrieval (with sub-query capture) ---
    seniority = _infer_seniority(jd.min_experience_years)
    industry = _infer_industry(jd.description, jd.company)

    try:
        subqueries = await _generate_subqueries(jd.job_title, seniority, industry)
    except Exception:
        subqueries = []
    if len(subqueries) < 3:
        subqueries = (subqueries + _fallback_subqueries(jd.job_title, seniority, industry))[:3]

    store = get_store()
    top_k = 5
    try:
        raw_results = await asyncio.gather(
            *[_retrieve_one(q, store, top_k * 2) for q in subqueries]
        )
    except Exception as exc:
        logger.warning("eval_retrieval_failed", case=name, reason=str(exc))
        return CaseResult(
            name=name,
            context_relevance=0.0,
            faithfulness=0.0,
            n_chunks_retrieved=0,
            n_claims_checked=0,
            subqueries=subqueries,
        )

    # --- Context relevance ---
    context_relevance = await compute_context_relevance(subqueries, list(raw_results))

    # --- Full pipeline for scorer rationales ---
    rag_context, _ = await retrieve_context(jd)
    parsed = await parse_resume(resume_text)
    score_output = await score_resume(jd=jd, parsed_resume=parsed, rag_context=rag_context)

    rationales = [ds.rationale for ds in score_output.dimension_scores.values()]

    # --- Faithfulness (checked against resume + JD + RAG context) ---
    grounding = _build_grounding(jd, parsed, rag_context)
    faithfulness = await compute_faithfulness(rationales, grounding)

    total_chunks = sum(len(r) for r in raw_results)

    logger.info(
        "eval_case_done",
        case=name,
        context_relevance=round(context_relevance, 3),
        faithfulness=round(faithfulness, 3),
        tier=score_output.tier.value,
    )

    return CaseResult(
        name=name,
        context_relevance=context_relevance,
        faithfulness=faithfulness,
        n_chunks_retrieved=total_chunks,
        n_claims_checked=0,
        subqueries=subqueries,
        tier=score_output.tier.value,
        composite_score=score_output.composite_score,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _print_report(results: list[CaseResult]) -> None:
    col_w = 42
    print("\n" + "=" * 80)
    print("  RecruitSense RAG Evaluation Report")
    print("=" * 80)
    header = f"{'Test Case':<{col_w}} {'Ctx Relevance':>14} {'Faithfulness':>13} {'Tier':>5} {'Score':>6}"
    print(header)
    print("-" * 80)
    for r in results:
        row = (
            f"{r.name[:col_w - 1]:<{col_w}}"
            f" {r.context_relevance:>13.3f}"
            f" {r.faithfulness:>13.3f}"
            f" {r.tier:>5}"
            f" {r.composite_score:>6.1f}"
        )
        print(row)
    print("-" * 80)
    avg_cr = sum(r.context_relevance for r in results) / len(results)
    avg_f = sum(r.faithfulness for r in results) / len(results)
    print(f"{'AVERAGE':<{col_w}} {avg_cr:>13.3f} {avg_f:>13.3f}")
    print("=" * 80)
    print()
    print("Metric definitions:")
    print("  Context Relevance — mean LLM-rated relevance of retrieved chunks to sub-queries (0–1)")
    print(
        "  Faithfulness      — fraction of scorer rationale claims grounded in resume + JD + RAG context (0–1)"
    )
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def _main(output_path: str | None, max_cases: int, delay: float) -> None:
    cases = _TEST_CASES[:max_cases] if max_cases > 0 else _TEST_CASES
    results = []
    for i, case in enumerate(cases):
        try:
            result = await evaluate_case(case)
        except Exception as exc:
            # A hard failure on one case (e.g. persistent 429) shouldn't sink
            # the whole report — record a zeroed row and continue.
            logger.warning("eval_case_crashed", case=case["name"], reason=str(exc))
            result = CaseResult(
                name=case["name"],
                context_relevance=0.0,
                faithfulness=0.0,
                n_chunks_retrieved=0,
                n_claims_checked=0,
            )
        results.append(result)
        # Cool-off between cases so free-tier per-minute limits can recover.
        if delay > 0 and i < len(cases) - 1:
            logger.info("eval_cooldown", seconds=delay)
            await asyncio.sleep(delay)

    _print_report(results)

    if output_path:
        report = [
            {
                "name": r.name,
                "context_relevance": round(r.context_relevance, 4),
                "faithfulness": round(r.faithfulness, 4),
                "n_chunks_retrieved": r.n_chunks_retrieved,
                "subqueries": r.subqueries,
                "tier": r.tier,
                "composite_score": r.composite_score,
            }
            for r in results
        ]
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate RAG context relevance and faithfulness.")
    parser.add_argument("--output", default=None, help="Save JSON report to this path.")
    parser.add_argument(
        "--max-cases",
        type=int,
        default=0,
        help="Evaluate only the first N test cases (0 = all). Use 1 to validate under rate limits.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to wait between cases so free-tier per-minute limits can recover.",
    )
    args = parser.parse_args()
    asyncio.run(_main(args.output, args.max_cases, args.delay))

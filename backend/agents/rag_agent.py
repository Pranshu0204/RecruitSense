"""RAG agent — RAG Fusion (3 sub-queries + Reciprocal Rank Fusion) over Qdrant.

Algorithm (per Ilin / RAG-Fusion 2023):
1. Infer ``seniority`` from JD's ``min_experience_years`` and ``industry`` from
   description/company keywords.
2. Ask the LLM (cached) to expand the JD into exactly 3 retrieval sub-queries
   covering: required skills, typical experience profile, common gaps.
3. Embed each sub-query with BGE-large; retrieve top 2k chunks from Qdrant in
   parallel.
4. Merge the three ranked lists with Reciprocal Rank Fusion
   (``RRF(d) = Σ 1 / (k + rank_q(d))``, k=60) and return the top-``k``.
5. Format the final chunks into a single context string for the scorer prompt.

Returns ``("", [])`` on any retrieval failure so the scorer can still proceed
without context (degraded but functional).
"""

from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any

from backend.agents import get_llm
from backend.core.prompts import RAG_SUBQUERY_PROMPT
from backend.core.schemas import JDInput, ParsedResume
from backend.rag.embedder import embed_query
from backend.rag.vector_store import QdrantStore, RetrievedChunk, get_store
from backend.utils.logger import get_logger
from backend.utils.redis_cache import cache_llm

logger = get_logger(__name__)


# --- JD enrichment heuristics -------------------------------------------------


def _infer_seniority(min_experience_years: float) -> str:
    """Map a numeric experience floor to a coarse seniority label."""
    if min_experience_years >= 5:
        return "senior"
    if min_experience_years >= 2:
        return "mid"
    return "junior"


_INDUSTRY_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fintech": ("fintech", "payments", "banking", "trading", "crypto"),
    "healthcare": ("healthcare", "clinical", "medical", "hospital", "biotech"),
    "e-commerce": ("e-commerce", "ecommerce", "retail", "marketplace"),
    "SaaS": ("saas", "b2b", "enterprise software"),
    "AI/ML": ("ai company", "ml platform", "machine learning startup"),
    "edtech": ("edtech", "education", "learning platform"),
    "gaming": ("game", "gaming", "studio"),
    "automotive": ("automotive", "autonomous vehicle", "self-driving"),
    "media": ("streaming", "video", "media"),
}


def _infer_industry(jd_description: str, company: str) -> str:
    """Lightweight keyword-based industry inference; returns ``"general technology"`` as fallback."""
    text = f"{jd_description} {company}".lower()
    for industry, keywords in _INDUSTRY_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            return industry
    return "general technology"


# --- Sub-query generation -----------------------------------------------------


@cache_llm(namespace="rag_subqueries")
async def _generate_subqueries(job_title: str, seniority: str, industry: str) -> list[str]:
    """Use the LLM to produce 3 retrieval sub-queries (cached)."""
    llm = get_llm(temperature=0.3)
    prompt = RAG_SUBQUERY_PROMPT.format_messages(
        job_title=job_title, seniority=seniority, industry=industry
    )
    response = await llm.ainvoke(prompt)
    raw_lines = str(response.content).splitlines()
    queries = [
        line.strip(" -•\t").lstrip("0123456789. ").strip()
        for line in raw_lines
    ]
    queries = [q for q in queries if q]
    return queries[:3]


def _fallback_subqueries(job_title: str, seniority: str, industry: str) -> list[str]:
    """Deterministic fallback sub-queries used when the LLM call fails."""
    return [
        f"{job_title} required skills benchmark {industry}",
        f"typical experience for {seniority} {job_title}",
        f"{job_title} common skill gaps entry vs senior",
    ]


# --- Reciprocal Rank Fusion ---------------------------------------------------


def reciprocal_rank_fusion(
    results_per_query: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    """Merge multiple ranked lists with RRF.

    ``RRF(d) = Σ_q 1 / (k + rank_q(d))`` where ``rank`` is 1-indexed.
    """
    scores: dict[str, float] = defaultdict(float)
    chunks_by_text: dict[str, RetrievedChunk] = {}

    for results in results_per_query:
        for rank, chunk in enumerate(results, start=1):
            scores[chunk.text] += 1.0 / (k + rank)
            chunks_by_text.setdefault(chunk.text, chunk)

    sorted_texts = sorted(scores, key=lambda t: scores[t], reverse=True)
    return [chunks_by_text[t] for t in sorted_texts]


# --- Retrieval helpers --------------------------------------------------------


async def _retrieve_one(query: str, store: QdrantStore, k: int) -> list[RetrievedChunk]:
    """Embed + search a single sub-query off the event loop."""
    vector = await asyncio.to_thread(embed_query, query)
    return await asyncio.to_thread(store.search, vector, k)


# --- Public entry point -------------------------------------------------------


async def retrieve_context(
    jd: JDInput,
    parsed_resume: ParsedResume | None = None,  # noqa: ARG001 — kept for future signal use
    top_k: int = 5,
) -> tuple[str, list[RetrievedChunk]]:
    """Run the full RAG-Fusion pipeline for a single JD.

    Args:
        jd: validated job description.
        parsed_resume: currently unused; reserved for future personalization
            of sub-queries by candidate background.
        top_k: number of fused chunks to return.

    Returns:
        ``(formatted_context_string, raw_chunks)``. ``("", [])`` on failure.
    """
    seniority = _infer_seniority(jd.min_experience_years)
    industry = _infer_industry(jd.description, jd.company)

    try:
        subqueries: list[str] = await _generate_subqueries(jd.job_title, seniority, industry)
    except Exception as exc:
        logger.warning("rag_subquery_generation_failed", reason=str(exc))
        subqueries = []

    if len(subqueries) < 3:
        subqueries = (subqueries + _fallback_subqueries(jd.job_title, seniority, industry))[:3]

    store = get_store()
    try:
        results_per_query = await asyncio.gather(
            *[_retrieve_one(q, store, top_k * 2) for q in subqueries]
        )
    except Exception as exc:
        logger.warning("rag_search_failed", reason=str(exc))
        return "", []

    fused = reciprocal_rank_fusion(results_per_query)[:top_k]
    formatted = "\n\n---\n\n".join(f"[score={c.score:.3f}] {c.text}" for c in fused)
    logger.info(
        "rag_retrieved", n_subqueries=len(subqueries), n_returned=len(fused),
        seniority=seniority, industry=industry,
    )
    return formatted, fused


# --- Internal exports for tests ----------------------------------------------

__all__ = [
    "retrieve_context",
    "reciprocal_rank_fusion",
    "_infer_seniority",
    "_infer_industry",
]


# Re-bind helpers for tests without leading underscore (private-ish public API)
infer_seniority = _infer_seniority
infer_industry = _infer_industry

"""RAG-Fusion ranking + JD-enrichment heuristics."""

from __future__ import annotations

import pytest

from backend.agents.rag_agent import (
    infer_industry,
    infer_seniority,
    reciprocal_rank_fusion,
)
from backend.rag.vector_store import RetrievedChunk


def _chunk(text: str, score: float = 0.5) -> RetrievedChunk:
    return RetrievedChunk(text=text, score=score, metadata={})


def test_rrf_promotes_chunks_appearing_in_multiple_lists() -> None:
    """A chunk in all 3 lists outranks a chunk only in one, even at high rank."""
    common = _chunk("python fastapi async")
    only_a = _chunk("kubernetes terraform")
    only_b = _chunk("machine learning pipelines")

    fused = reciprocal_rank_fusion(
        [
            [common, only_a],
            [common, only_b],
            [common],
        ]
    )
    assert fused[0].text == common.text


def test_rrf_handles_empty_lists() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


def test_rrf_dedupes_by_text() -> None:
    """The same chunk text appearing twice must collapse into one output entry."""
    a = _chunk("same text", score=0.9)
    b = _chunk("same text", score=0.4)
    fused = reciprocal_rank_fusion([[a], [b]])
    assert len(fused) == 1


@pytest.mark.parametrize(
    ("years", "expected"),
    [(0.0, "junior"), (1.5, "junior"), (2.0, "mid"), (4.9, "mid"), (5.0, "senior"), (10.0, "senior")],
)
def test_infer_seniority(years: float, expected: str) -> None:
    assert infer_seniority(years) == expected


@pytest.mark.parametrize(
    ("desc", "company", "expected_substr"),
    [
        ("Crypto trading platform", "", "fintech"),
        ("Clinical trials data", "Acme Health", "healthcare"),
        ("Online retail marketplace", "", "e-commerce"),
        ("Just a generic web app", "", "general"),
    ],
)
def test_infer_industry(desc: str, company: str, expected_substr: str) -> None:
    assert expected_substr in infer_industry(desc, company)

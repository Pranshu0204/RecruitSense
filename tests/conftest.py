"""Shared pytest fixtures.

The HTTP-level tests mount the real FastAPI app but monkeypatch the boundary
functions (``run_pipeline``, vector-store ``health()``, Redis ping) so the
suite never reaches OpenRouter, Qdrant, or Redis. This keeps CI hermetic and
deterministic — and lets a contributor run ``pytest`` on a fresh clone with
no infra running locally.
"""

import os
from collections.abc import Iterator

import pytest

# Set env vars BEFORE importing the app so Settings picks them up.
os.environ.setdefault("OPENROUTER_API_KEY", "test-key")
os.environ.setdefault("QDRANT_HOST", "localhost")
os.environ.setdefault("REDIS_HOST", "localhost")
os.environ.setdefault("LOG_LEVEL", "WARNING")


@pytest.fixture
def client() -> Iterator:
    """FastAPI ``TestClient`` against the production app."""
    from fastapi.testclient import TestClient

    from backend.api.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_jd_payload() -> dict:
    """Minimal valid JD JSON for ``/screen`` and ``/batch`` requests."""
    return {
        "job_title": "Senior Python Engineer",
        "company": "Acme",
        "description": "Looking for a senior Python engineer with FastAPI experience.",
        "required_skills": ["python", "fastapi"],
        "preferred_skills": ["kubernetes"],
        "min_experience_years": 5.0,
        "education_level": "bachelor",
    }


@pytest.fixture
def sample_score_output_dict() -> dict:
    """A valid ``ScoreOutput`` payload (used as the mocked pipeline return)."""
    return {
        "candidate_name": "Test Candidate",
        "composite_score": 78.5,
        "tier": "B",
        "dimension_scores": {
            "skills_match": {"score": 8.0, "rationale": "Strong overlap with required stack."},
            "experience_relevance": {"score": 8.5, "rationale": "Years of senior Python work."},
            "education_and_certs": {"score": 7.0, "rationale": "Holds a bachelor's degree."},
            "project_impact": {"score": 7.5, "rationale": "Multiple shipped projects."},
            "communication_and_polish": {"score": 7.0, "rationale": "Resume well structured."},
        },
        "top_strengths": ["Deep Python expertise", "FastAPI in production"],
        "key_gaps": ["No Kubernetes experience"],
        "bias_flags": [],
        "recommended_action": "interview",
        "rag_context_used": "[score=0.92] Senior Python Engineer requires FastAPI...",
        "confidence": 0.88,
        "model_used": "test-model",
    }


@pytest.fixture
def tiny_pdf_bytes() -> bytes:
    """A minimal real PDF generated in-memory by PyMuPDF for upload tests."""
    import fitz

    doc = fitz.open()
    page = doc.new_page()
    page.insert_text(
        (72, 72),
        "Jane Smith\nSenior Python Engineer\n5 years FastAPI experience\nBSc Computer Science",
    )
    pdf_bytes: bytes = doc.tobytes()
    doc.close()
    return pdf_bytes

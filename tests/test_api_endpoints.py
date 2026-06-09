"""HTTP-level tests for ``/health``, ``/screen``, and ``/batch``.

External dependencies (the LLM pipeline and the vector / cache stores) are
monkeypatched at import-resolution boundaries so the suite stays hermetic.
"""

import json

import pytest

from backend.core.schemas import ScoreOutput


# --- /health -----------------------------------------------------------------
def test_health_ok_when_both_deps_up(monkeypatch: pytest.MonkeyPatch, client) -> None:
    """Both deps healthy → ``status: ok`` and 200."""
    from backend.api.routes import health as health_route

    class _Stub:
        def health(self) -> bool:
            return True

    monkeypatch.setattr(health_route, "get_store", lambda: _Stub())
    monkeypatch.setattr(health_route, "is_redis_available", lambda: True)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body == {"status": "ok", "qdrant": True, "redis": True}


def test_health_degraded_when_redis_down(monkeypatch: pytest.MonkeyPatch, client) -> None:
    """One dep down → ``status: degraded`` but still 200 (so monitors can read it)."""
    from backend.api.routes import health as health_route

    class _Stub:
        def health(self) -> bool:
            return True

    monkeypatch.setattr(health_route, "get_store", lambda: _Stub())
    monkeypatch.setattr(health_route, "is_redis_available", lambda: False)

    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "degraded"
    assert body["redis"] is False


# --- /screen -----------------------------------------------------------------
def test_screen_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    client,
    sample_jd_payload: dict,
    sample_score_output_dict: dict,
    tiny_pdf_bytes: bytes,
) -> None:
    """End-to-end multipart POST with mocked pipeline returns the ScoreOutput JSON."""
    from backend.api.routes import screen as screen_route

    async def fake_pipeline(_jd, _resume_text, model=""):
        return ScoreOutput.model_validate(sample_score_output_dict)

    monkeypatch.setattr(screen_route, "run_pipeline", fake_pipeline)

    resp = client.post(
        "/screen",
        data={"jd_json": json.dumps(sample_jd_payload)},
        files={"resume": ("cv.pdf", tiny_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["candidate_name"] == "Test Candidate"
    assert body["tier"] == "B"
    assert body["composite_score"] == 78.5


def test_screen_rejects_non_pdf(client, sample_jd_payload: dict, tiny_pdf_bytes: bytes) -> None:
    """Files without a ``.pdf`` extension must be rejected with 400."""
    resp = client.post(
        "/screen",
        data={"jd_json": json.dumps(sample_jd_payload)},
        files={"resume": ("cv.docx", tiny_pdf_bytes, "application/octet-stream")},
    )
    assert resp.status_code == 400


def test_screen_rejects_invalid_jd(client, tiny_pdf_bytes: bytes) -> None:
    """A JD that fails Pydantic validation must come back as 422."""
    bad_jd = {"job_title": "", "description": "short"}
    resp = client.post(
        "/screen",
        data={"jd_json": json.dumps(bad_jd)},
        files={"resume": ("cv.pdf", tiny_pdf_bytes, "application/pdf")},
    )
    assert resp.status_code == 422


def test_screen_rejects_empty_pdf(client, sample_jd_payload: dict) -> None:
    """Empty upload → 400 with a clear error message."""
    resp = client.post(
        "/screen",
        data={"jd_json": json.dumps(sample_jd_payload)},
        files={"resume": ("cv.pdf", b"", "application/pdf")},
    )
    assert resp.status_code == 400


# --- /batch ------------------------------------------------------------------
def test_batch_happy_path(
    monkeypatch: pytest.MonkeyPatch,
    client,
    sample_jd_payload: dict,
    sample_score_output_dict: dict,
    tiny_pdf_bytes: bytes,
) -> None:
    """Multi-PDF batch returns a sorted leaderboard + tier distribution."""
    from backend.api.routes import batch as batch_route

    async def fake_pipeline(_jd, _resume_text, model=""):
        return ScoreOutput.model_validate(sample_score_output_dict)

    monkeypatch.setattr(batch_route, "run_pipeline", fake_pipeline)

    files = [("resumes", (f"cv_{i}.pdf", tiny_pdf_bytes, "application/pdf")) for i in range(3)]
    resp = client.post(
        "/batch",
        data={"jd_json": json.dumps(sample_jd_payload)},
        files=files,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["total_resumes"] == 3
    assert len(body["ranked_candidates"]) == 3
    # Tier B counted in distribution; shortlisted should match
    assert body["tier_distribution"]["B"] == 3
    assert body["shortlisted_count"] == 3


def test_batch_rejects_no_resumes(client, sample_jd_payload: dict) -> None:
    """Missing ``resumes`` field must come back as a validation error."""
    resp = client.post(
        "/batch",
        data={"jd_json": json.dumps(sample_jd_payload)},
    )
    assert resp.status_code in (400, 422)


def test_batch_failed_resume_does_not_kill_batch(
    monkeypatch: pytest.MonkeyPatch,
    client,
    sample_jd_payload: dict,
    sample_score_output_dict: dict,
    tiny_pdf_bytes: bytes,
) -> None:
    """An empty/invalid PDF in a batch should degrade to a zero-confidence row."""
    from backend.api.routes import batch as batch_route

    async def fake_pipeline(_jd, _resume_text, model=""):
        return ScoreOutput.model_validate(sample_score_output_dict)

    monkeypatch.setattr(batch_route, "run_pipeline", fake_pipeline)

    files = [
        ("resumes", ("good.pdf", tiny_pdf_bytes, "application/pdf")),
        ("resumes", ("empty.pdf", b"", "application/pdf")),
        ("resumes", ("good2.pdf", tiny_pdf_bytes, "application/pdf")),
    ]
    resp = client.post("/batch", data={"jd_json": json.dumps(sample_jd_payload)}, files=files)
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_resumes"] == 3
    # The two good ones land in tier B; the failed empty PDF should be tier D.
    assert body["tier_distribution"]["B"] == 2
    assert body["tier_distribution"]["D"] == 1
    assert body["shortlisted_count"] == 2

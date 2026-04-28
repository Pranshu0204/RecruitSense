"""Versioned LLM prompt templates for RecruitSense agents.

Templates are LangChain ``ChatPromptTemplate`` instances so they can be invoked
directly by ``LLMChain`` / ``RunnableSequence`` callers in Phase 5.

Bumping ``PROMPT_VERSION`` invalidates the Redis cache (the cache key includes
the version), so prompt changes never silently serve stale completions.
"""

from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

PROMPT_VERSION: str = "v1"


# --- Parser agent -------------------------------------------------------------

_PARSER_SYSTEM = """You are an expert resume parser. Extract structured data from raw resume text and return ONLY valid JSON matching the requested schema.

Rules:
- Infer implicit skills from project descriptions (e.g., "built a REST API in Flask" implies REST, HTTP, Flask, API design).
- Compute total_years_experience by summing role durations; if dates are missing or partial, estimate conservatively.
- Do NOT invent skills the resume does not support.
- Output a single JSON object. No prose. No markdown. No code fences."""

_PARSER_USER = """Resume text:
\"\"\"
{resume_text}
\"\"\"

Return JSON with these exact keys:
- candidate_name (str)
- skills (list[str])
- experience (list[str], one role per item formatted as "Title @ Company (start - end)")
- education (list[str])
- certifications (list[str])
- projects (list[str])
- total_years_experience (float)"""

PARSER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", _PARSER_SYSTEM), ("user", _PARSER_USER)]
)


# --- RAG sub-query generator --------------------------------------------------

_RAG_SUBQUERY_SYSTEM = """You expand a job-matching task into exactly 3 retrieval queries for a skills/role knowledge base. Output the 3 queries on separate lines, one per line, no numbering, no commentary."""

_RAG_SUBQUERY_USER = """Job role: {job_title}
Seniority hint: {seniority}
Industry: {industry}

Generate three retrieval queries covering:
1. Required skills for this role in this industry.
2. Typical experience profile for this seniority level.
3. Common skill gaps separating entry-level from senior in this role."""

RAG_SUBQUERY_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", _RAG_SUBQUERY_SYSTEM), ("user", _RAG_SUBQUERY_USER)]
)


# --- Scorer agent -------------------------------------------------------------

_SCORER_SYSTEM = """You are an expert resume scorer for a recruiter. Score the candidate against the JD across five dimensions:

1. skills_match (weight 0.35) — overlap with required & preferred skills.
2. experience_relevance (weight 0.30) — years and domain alignment with the JD.
3. education_and_certs (weight 0.15) — degree level + relevant certifications.
4. project_impact (weight 0.10) — measurable outcomes, complexity, ownership.
5. communication_and_polish (weight 0.10) — clarity, structure, professionalism of resume.

Process:
- First emit a <thinking>...</thinking> block with step-by-step reasoning.
- Then, OUTSIDE the thinking block, emit a single JSON object and nothing else.

Each dimension score is a float in [0.0, 10.0] with a one-sentence rationale.
Bias-correlated factors (gender, age, ethnicity, nationality, address) MUST NOT influence any score."""

_SCORER_USER = """Job Description
---------------
Title: {job_title}
Company: {company}
Required skills: {required_skills}
Preferred skills: {preferred_skills}
Minimum experience: {min_experience_years} years
Education level: {education_level}
Description:
{jd_description}

Parsed Resume (JSON)
--------------------
{parsed_resume_json}

Knowledge-base Context (skills taxonomy & role benchmarks from RAG)
-------------------------------------------------------------------
{rag_context}

Return JSON with this exact shape:
{{
  "dimension_scores": {{
    "skills_match":             {{"score": <float 0-10>, "rationale": "<one sentence>"}},
    "experience_relevance":     {{"score": <float 0-10>, "rationale": "<one sentence>"}},
    "education_and_certs":      {{"score": <float 0-10>, "rationale": "<one sentence>"}},
    "project_impact":           {{"score": <float 0-10>, "rationale": "<one sentence>"}},
    "communication_and_polish": {{"score": <float 0-10>, "rationale": "<one sentence>"}}
  }},
  "top_strengths": ["<string>", ...],
  "key_gaps": ["<string>", ...],
  "confidence": <float 0-1>
}}"""

SCORER_PROMPT: ChatPromptTemplate = ChatPromptTemplate.from_messages(
    [("system", _SCORER_SYSTEM), ("user", _SCORER_USER)]
)


__all__ = [
    "PROMPT_VERSION",
    "PARSER_PROMPT",
    "RAG_SUBQUERY_PROMPT",
    "SCORER_PROMPT",
]

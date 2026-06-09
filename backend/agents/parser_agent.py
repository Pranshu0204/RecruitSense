"""Parser agent — extracts a structured ``ParsedResume`` from raw resume text.

Uses a cached LLM call, then augments the result with regex-based implicit skill
inference and job-title normalization before Pydantic validation. Falls back to
a near-empty ParsedResume on any error so a single bad file can't break a batch.
"""

import json
import re
from typing import Any

from backend.agents import get_llm
from backend.core.prompts import PARSER_PROMPT
from backend.core.schemas import ParsedResume
from backend.utils.logger import get_logger
from backend.utils.redis_cache import cache_llm

logger = get_logger(__name__)
# --- Job-title normalization taxonomy -----------------------------------------

JOB_TITLE_TAXONOMY: dict[str, str] = {
    "sde": "Software Engineer",
    "sde 1": "Software Engineer",
    "sde 2": "Software Engineer",
    "sde i": "Software Engineer",
    "sde ii": "Software Engineer",
    "swe": "Software Engineer",
    "be": "Backend Engineer",
    "fe": "Frontend Engineer",
    "fs": "Full-Stack Engineer",
    "fullstack": "Full-Stack Engineer",
    "ml": "ML Engineer",
    "ml eng": "ML Engineer",
    "ds": "Data Scientist",
    "de": "Data Engineer",
    "data eng": "Data Engineer",
    "devops": "DevOps Engineer",
    "sre": "Site Reliability Engineer",
    "qa": "QA Engineer",
    "qae": "QA Engineer",
    "sdet": "Test Automation Engineer",
    "pm": "Product Manager",
    "tpm": "Technical Product Manager",
    "em": "Engineering Manager",
    "tl": "Tech Lead",
    "vp eng": "VP of Engineering",
    "cto": "Chief Technology Officer",
    "ios eng": "Mobile iOS Developer",
    "android eng": "Mobile Android Developer",
    "rn dev": "React Native Developer",
    "cv eng": "Computer Vision Engineer",
    "nlp eng": "NLP Engineer",
    "llm eng": "LLM Engineer",
}


def normalize_job_title(title: str) -> str:
    """Normalize an abbreviated/lowercase job title to a canonical form."""
    cleaned = title.strip().lower().rstrip(".,")
    return JOB_TITLE_TAXONOMY.get(cleaned, title.strip())


# --- Implicit-skill inference -------------------------------------------------

_IMPLICIT_SKILL_PATTERNS: tuple[tuple[re.Pattern[str], tuple[str, ...]], ...] = (
    (re.compile(r"\brest\s*ap[is]?\b", re.IGNORECASE), ("REST", "HTTP", "API design")),
    (re.compile(r"\bgraphql\b", re.IGNORECASE), ("GraphQL", "API design")),
    (re.compile(r"\bgrpc\b", re.IGNORECASE), ("gRPC", "Protocol Buffers")),
    (re.compile(r"\bmicroservice", re.IGNORECASE), ("Microservices", "Distributed systems")),
    (re.compile(r"\bk8s\b|\bkubernetes\b", re.IGNORECASE), ("Kubernetes", "Containers")),
    (re.compile(r"\bdocker\b", re.IGNORECASE), ("Docker", "Containers")),
    (re.compile(r"\bci[/\s]*cd\b", re.IGNORECASE), ("CI/CD",)),
    (re.compile(r"\bterraform\b", re.IGNORECASE), ("Terraform", "Infrastructure-as-code")),
    (re.compile(r"\baws\b|\bec2\b|\bs3\b|\blambda\b", re.IGNORECASE), ("AWS",)),
    (re.compile(r"\bgcp\b|\bbigquery\b", re.IGNORECASE), ("GCP",)),
    (re.compile(r"\bazure\b", re.IGNORECASE), ("Azure",)),
    (re.compile(r"\bpostgres", re.IGNORECASE), ("PostgreSQL", "SQL")),
    (re.compile(r"\bmysql\b", re.IGNORECASE), ("MySQL", "SQL")),
    (re.compile(r"\bredis\b", re.IGNORECASE), ("Redis", "Caching")),
    (re.compile(r"\bkafka\b", re.IGNORECASE), ("Kafka", "Streaming")),
    (re.compile(r"\bspark\b", re.IGNORECASE), ("Apache Spark", "Distributed processing")),
    (re.compile(r"\bairflow\b", re.IGNORECASE), ("Apache Airflow", "Workflow orchestration")),
    (re.compile(r"\bdbt\b", re.IGNORECASE), ("dbt", "Data modeling")),
    (re.compile(r"\bpytorch\b", re.IGNORECASE), ("PyTorch",)),
    (re.compile(r"\btensorflow\b|\btf\b", re.IGNORECASE), ("TensorFlow",)),
    (re.compile(r"\btransformer", re.IGNORECASE), ("Transformers", "Deep learning")),
    (re.compile(r"\brag\b|\bretrieval[- ]augmented", re.IGNORECASE), ("RAG", "Vector databases")),
    (re.compile(r"\blangchain\b", re.IGNORECASE), ("LangChain", "LLM orchestration")),
    (re.compile(r"\bnext\.?js\b", re.IGNORECASE), ("Next.js", "React")),
    (re.compile(r"\bfastapi\b", re.IGNORECASE), ("FastAPI", "Python")),
    (re.compile(r"\bdjango\b", re.IGNORECASE), ("Django", "Python")),
    (re.compile(r"\breact\s*native\b", re.IGNORECASE), ("React Native", "Mobile development")),
)


def infer_implicit_skills(text: str, existing_skills: list[str]) -> list[str]:
    """Return new skills implied by ``text`` (case-insensitive, dedup'd vs existing)."""
    if not text:
        return []
    inferred: set[str] = set()
    existing_lower = {s.strip().lower() for s in existing_skills}
    for pattern, skills in _IMPLICIT_SKILL_PATTERNS:
        if pattern.search(text):
            for s in skills:
                if s.lower() not in existing_lower:
                    inferred.add(s)
    return sorted(inferred)


# --- Output cleaning ----------------------------------------------------------

_FENCE_OPEN = re.compile(r"^```(?:json)?\s*\n?", re.IGNORECASE)
_FENCE_CLOSE = re.compile(r"\n?```\s*$")


def _strip_code_fences(text: str) -> str:
    """Remove leading/trailing markdown code fences from LLM output."""
    text = text.strip()
    if text.startswith("```"):
        text = _FENCE_OPEN.sub("", text)
        text = _FENCE_CLOSE.sub("", text)
    return text.strip()


def _normalize_experience_titles(experiences: list[str]) -> list[str]:
    """Normalize the title portion of each ``"Title @ Company (...)"`` entry."""
    normalized: list[str] = []
    for exp in experiences:
        parts = re.split(r"\s+@\s+|\s+at\s+", exp, maxsplit=1)
        if len(parts) == 2:
            normalized.append(f"{normalize_job_title(parts[0])} @ {parts[1]}")
        else:
            normalized.append(exp)
    return normalized


# --- LLM call -----------------------------------------------------------------
@cache_llm(namespace="parser")
async def _parser_llm_call(resume_text: str) -> dict[str, Any]:
    """Cached LLM call. Returns a parsed JSON dict (raises on failure)."""
    llm = get_llm()
    prompt = PARSER_PROMPT.format_messages(resume_text=resume_text)
    response = await llm.ainvoke(prompt)
    content = _strip_code_fences(str(response.content))
    return json.loads(content)


# --- Public entry point -------------------------------------------------------
async def parse_resume(resume_text: str) -> ParsedResume:
    """Parse raw resume text into a ``ParsedResume``.

    On any failure (LLM error, JSON parse error, validation error) returns a
    near-empty ``ParsedResume`` with ``candidate_name="Unknown"`` so a single
    bad resume cannot break a batch run.
    """
    if not resume_text or not resume_text.strip():
        return ParsedResume(candidate_name="Unknown")

    try:
        raw = await _parser_llm_call(resume_text)
    except Exception as exc:
        logger.warning("parser_llm_failed", reason=str(exc))
        return ParsedResume(candidate_name="Unknown")

    # Implicit-skill augmentation from project text.
    project_text = " ".join(raw.get("projects", []) or [])
    existing_skills = list(raw.get("skills") or [])
    if project_text:
        implicit = infer_implicit_skills(project_text, existing_skills)
        existing_skills = sorted(set(existing_skills) | set(implicit))
    raw["skills"] = existing_skills

    # Normalize job titles in experience entries.
    if raw.get("experience"):
        raw["experience"] = _normalize_experience_titles(list(raw["experience"]))

    try:
        return ParsedResume(**raw)
    except Exception as exc:
        logger.warning("parser_validation_failed", reason=str(exc), raw_keys=list(raw.keys()))
        return ParsedResume(
            candidate_name=str(raw.get("candidate_name") or "Unknown"),
            skills=existing_skills,
        )

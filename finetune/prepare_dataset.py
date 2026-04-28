"""Convert ``AzharAli05/Resume-Screening-Dataset`` into instruction-tuning JSONL.

The HF dataset has a ``Resume`` text column and a ``Category`` label (the broad
job family — "Data Science", "DevOps Engineer", etc.). It does NOT ship
ground-truth scoring outputs, so this script:

1. Loads the dataset from the HuggingFace Hub (cached locally).
2. For each row, synthesizes a plausible JD for the given category from a
   handful of templates.
3. Generates a deterministic *target* JSON in the exact ``ScoreOutput`` shape
   the production scorer emits, with per-dimension scores derived from a
   simple keyword-overlap heuristic.
4. Writes train / val splits as chat-formatted JSONL ready for
   :func:`trl.SFTTrainer` (one ``messages`` list per line).

The resulting dataset teaches the base model the *shape* and *style* of the
scoring output — it is an instruction-format conditioner, not a ground-truth
distillation. Use a stronger teacher LLM if you want to align scores to a
reference judge.

Usage::

    python -m finetune.prepare_dataset --max-samples 2000 --val-frac 0.1
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Any

from datasets import load_dataset

from backend.core.schemas import (
    DIMENSION_NAMES,
    Tier,
    composite_from_dimensions,
    tier_from_composite,
)
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DATASET_ID = "AzharAli05/Resume-Screening-Dataset"
OUTPUT_DIR = Path("finetune/dataset/data")

# A short, generic system prompt that mirrors the runtime SCORER_PROMPT shape
# without dragging the full prompt template (we want the model to learn the
# *output schema*, not memorize the exact wording of any one prompt).
SYSTEM_PROMPT = (
    "You are RecruitSense, an expert technical recruiter. Score a candidate "
    "against a job description across five dimensions: skills_match (35%), "
    "experience_relevance (30%), education_and_certs (15%), project_impact "
    "(10%), communication_and_polish (10%). Respond with a single JSON object "
    "matching the ScoreOutput schema — no prose, no markdown, no code fences."
)

# JD templates per coarse category. The placeholder ``{cat}`` is filled with
# the dataset's ``Category`` value verbatim so the model sees realistic phrasing.
JD_TEMPLATES = [
    "We are hiring a {cat} to join our growing engineering team. The ideal "
    "candidate has hands-on experience shipping production systems, strong "
    "fundamentals, and a track record of delivering measurable impact.",
    "Looking for a senior {cat} who can own end-to-end delivery, mentor "
    "junior engineers, and partner with product to translate business needs "
    "into technical roadmaps. Experience with cloud platforms preferred.",
    "Mid-level {cat} role on a fast-moving team. You will design, build, "
    "and operate services that power critical product surfaces. Strong "
    "collaboration, clean code, and pragmatic decision-making are essential.",
]

# Coarse keyword sets per common category. Used to score skills_match.
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Data Science": ["python", "pandas", "scikit-learn", "tensorflow", "pytorch", "sql", "ml", "statistics"],
    "DevOps Engineer": ["docker", "kubernetes", "terraform", "aws", "ci/cd", "linux", "ansible", "jenkins"],
    "Web Designing": ["html", "css", "javascript", "figma", "adobe", "ui", "ux", "responsive"],
    "Java Developer": ["java", "spring", "hibernate", "maven", "junit", "rest", "microservices", "jvm"],
    "Python Developer": ["python", "django", "flask", "fastapi", "rest", "sql", "celery", "pytest"],
    "Web Developer": ["javascript", "react", "node", "html", "css", "rest", "git", "typescript"],
    "Network Security Engineer": ["firewall", "vpn", "ids", "ips", "siem", "tcp", "linux", "wireshark"],
    "Mechanical Engineer": ["solidworks", "autocad", "ansys", "matlab", "cad", "fea", "manufacturing"],
    "Civil Engineer": ["autocad", "staad", "revit", "structural", "concrete", "construction"],
    "Electrical Engineer": ["matlab", "plc", "scada", "circuit", "autocad", "embedded"],
}
DEFAULT_KEYWORDS = ["python", "sql", "git", "linux", "rest", "docker", "aws"]

# Token used to detect a year of experience like "5 years" / "5+ years".
YEARS_RE = re.compile(r"(\d+)\+?\s+(?:year|yr)s?", re.IGNORECASE)


def _candidate_keywords(category: str) -> list[str]:
    """Pick the keyword set for the resume's coarse category, or a fallback."""
    return CATEGORY_KEYWORDS.get(category, DEFAULT_KEYWORDS)


def _coverage(resume_text: str, keywords: list[str]) -> float:
    """Fraction of ``keywords`` appearing in ``resume_text`` (case-insensitive)."""
    if not keywords:
        return 0.0
    text = resume_text.lower()
    hits = sum(1 for kw in keywords if kw.lower() in text)
    return hits / len(keywords)


def _years_of_experience(resume_text: str) -> float:
    """Extract a rough years-of-experience number from the resume body."""
    matches = YEARS_RE.findall(resume_text)
    if not matches:
        return 0.0
    try:
        return float(max(int(m) for m in matches))
    except ValueError:
        return 0.0


def _has_any(resume_text: str, terms: list[str]) -> bool:
    """True iff any term from ``terms`` appears in ``resume_text``."""
    text = resume_text.lower()
    return any(t.lower() in text for t in terms)


def synthesize_target(resume_text: str, category: str, rng: random.Random) -> dict[str, Any]:
    """Build a deterministic-ish ``ScoreOutput``-shaped target for this row.

    Scores are derived from cheap heuristics so the model learns the *output
    schema* and the *kind of rationale* expected — not absolute hiring truth.
    """
    keywords = _candidate_keywords(category)
    coverage = _coverage(resume_text, keywords)
    years = _years_of_experience(resume_text)
    has_degree = _has_any(
        resume_text, ["b.tech", "bachelor", "b.e.", "btech", "bsc", "ba ", "ms ", "m.tech", "msc", "mba", "phd"]
    )
    has_certs = _has_any(resume_text, ["certified", "certificate", "certification"])
    has_projects = _has_any(resume_text, ["project", "built", "developed", "designed", "implemented"])

    # 0–10 scaled scores
    skills = round(min(10.0, coverage * 10.0 + rng.uniform(-0.5, 0.5)), 2)
    exp = round(min(10.0, years * 1.2 + rng.uniform(-0.5, 1.0) + 3.0), 2) if years else round(rng.uniform(3.0, 5.5), 2)
    edu = round(rng.uniform(7.0, 9.5) if has_degree else rng.uniform(3.0, 6.0), 2)
    if has_certs:
        edu = round(min(10.0, edu + 0.7), 2)
    proj = round(rng.uniform(6.5, 9.0) if has_projects else rng.uniform(3.5, 6.0), 2)
    polish = round(min(10.0, max(3.0, len(resume_text) / 300.0 + rng.uniform(-0.5, 1.0))), 2)

    dim_scores = {
        "skills_match": {
            "score": skills,
            "rationale": (
                f"Resume references {int(coverage * len(keywords))} of "
                f"{len(keywords)} target keywords for the {category} role."
            ),
        },
        "experience_relevance": {
            "score": exp,
            "rationale": (
                f"Detected ~{int(years)} years of relevant experience" if years
                else "Years-of-experience signal not explicitly stated; inferred from role descriptions."
            ),
        },
        "education_and_certs": {
            "score": edu,
            "rationale": (
                "Holds a recognized degree" + (" with relevant certifications." if has_certs else ".")
                if has_degree
                else "No clearly stated degree or certifications detected."
            ),
        },
        "project_impact": {
            "score": proj,
            "rationale": (
                "Resume describes shipped projects with technical scope."
                if has_projects
                else "Limited project narrative — mostly responsibilities rather than outcomes."
            ),
        },
        "communication_and_polish": {
            "score": polish,
            "rationale": "Resume is reasonably structured and proportional to expected length.",
        },
    }

    # Build a typed view, then compute composite/tier deterministically.
    from backend.core.schemas import DimensionScore

    typed = {k: DimensionScore(**v) for k, v in dim_scores.items()}
    composite = composite_from_dimensions(typed)
    tier = tier_from_composite(composite)

    strengths = []
    if skills >= 7:
        strengths.append(f"Strong overlap with {category} core skills.")
    if exp >= 7:
        strengths.append("Solid years of relevant experience.")
    if proj >= 7:
        strengths.append("Track record of shipped projects.")
    if not strengths:
        strengths.append("Resume covers fundamentals at an entry-level depth.")

    gaps = []
    if skills < 6:
        gaps.append(f"Missing several keywords expected for a {category} role.")
    if not has_degree:
        gaps.append("No formal degree clearly stated.")
    if not has_projects:
        gaps.append("Few concrete project outcomes.")
    if not gaps:
        gaps.append("No critical gaps identified for this seniority band.")

    action = {
        Tier.A: "strong_hire",
        Tier.B: "interview",
        Tier.C: "maybe",
        Tier.D: "reject",
    }[tier]

    return {
        "candidate_name": "Candidate",
        "composite_score": composite,
        "tier": tier.value,
        "dimension_scores": dim_scores,
        "top_strengths": strengths[:5],
        "key_gaps": gaps[:5],
        "bias_flags": [],
        "recommended_action": action,
        "rag_context_used": "",
        "confidence": round(rng.uniform(0.7, 0.92), 2),
    }


def build_record(resume_text: str, category: str, rng: random.Random) -> dict[str, Any] | None:
    """Build a single chat-format training record. Returns ``None`` if invalid."""
    resume_text = (resume_text or "").strip()
    category = (category or "").strip() or "Software Engineer"
    if len(resume_text) < 200:
        return None  # too short to learn from

    jd_template = rng.choice(JD_TEMPLATES)
    jd = jd_template.format(cat=category)

    target = synthesize_target(resume_text, category, rng)
    user_msg = (
        f"## Job Description\n{jd}\n\n"
        f"## Candidate Resume\n{resume_text[:6000]}\n\n"
        "Score this candidate against the job description."
    )
    assistant_msg = json.dumps(target, ensure_ascii=False)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg},
        ],
        "category": category,
    }


def _detect_columns(ds) -> tuple[str, str]:
    """Pick the resume-text and category columns from the loaded dataset."""
    cols = set(ds.column_names)
    resume_col = next(
        (c for c in ("Resume", "resume", "Resume_str", "Resume_text", "text") if c in cols),
        None,
    )
    cat_col = next(
        (c for c in ("Category", "category", "label", "Job_Title") if c in cols), None
    )
    if resume_col is None:
        raise ValueError(f"Could not find a resume-text column in {sorted(cols)}")
    return resume_col, cat_col or resume_col


def main() -> None:
    """Build train/val JSONL files under ``finetune/dataset/data/``."""
    parser = argparse.ArgumentParser(description="Prepare instruction-tuning dataset.")
    parser.add_argument("--max-samples", type=int, default=2000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    logger.info("dataset_loading", dataset=DATASET_ID, split=args.split)
    ds = load_dataset(DATASET_ID, split=args.split)
    logger.info("dataset_loaded", rows=len(ds), columns=ds.column_names)

    resume_col, cat_col = _detect_columns(ds)

    indices = list(range(len(ds)))
    rng.shuffle(indices)
    if args.max_samples:
        indices = indices[: args.max_samples]

    records: list[dict[str, Any]] = []
    for i in indices:
        row = ds[i]
        rec = build_record(row[resume_col], row.get(cat_col, ""), rng)
        if rec is not None:
            records.append(rec)

    if not records:
        raise RuntimeError("No usable records produced — check dataset columns.")

    rng.shuffle(records)
    n_val = max(1, int(len(records) * args.val_frac))
    val_records = records[:n_val]
    train_records = records[n_val:]

    train_path = OUTPUT_DIR / "train.jsonl"
    val_path = OUTPUT_DIR / "val.jsonl"
    _write_jsonl(train_path, train_records)
    _write_jsonl(val_path, val_records)

    # Tier distribution sanity check
    tier_counts: dict[str, int] = {}
    for r in records:
        target = json.loads(r["messages"][-1]["content"])
        tier_counts[target["tier"]] = tier_counts.get(target["tier"], 0) + 1

    logger.info(
        "dataset_written",
        train=len(train_records),
        val=len(val_records),
        train_path=str(train_path),
        val_path=str(val_path),
        tier_distribution=tier_counts,
        dimensions=list(DIMENSION_NAMES),
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write one JSON object per line."""
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

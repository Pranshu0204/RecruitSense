"""One-shot ingestion of skill taxonomy + sample JD/resume pairs into Qdrant.

Run via::

    make ingest                       # idempotent — skips collection if it exists
    python -m backend.rag.ingest --recreate    # drop and rebuild

The taxonomy is read from ``data/skill_taxonomy.json`` and flattened into
multiple per-role chunks (overview, tech-stack, seniority benchmark, common
gap). Twenty short sample JD/resume match descriptions are appended inline so
the RAG agent always has grounded examples to retrieve.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from backend.rag.embedder import embed_documents
from backend.rag.vector_store import QdrantStore
from backend.utils.logger import get_logger

logger = get_logger(__name__)

DATA_DIR: Path = Path(__file__).resolve().parents[2] / "data"
TAXONOMY_PATH: Path = DATA_DIR / "skill_taxonomy.json"


# --- Twenty synthesized JD ↔ ideal-resume match descriptions ----------------
# Each is a short paragraph the RAG agent can retrieve for "what does a strong
# candidate look like for X role" prompts. Short, varied across roles/seniority.

SAMPLE_JD_RESUME_PAIRS: list[dict[str, Any]] = [
    {
        "text": (
            "JD: Senior Backend Engineer at a fintech requiring Python, FastAPI, "
            "PostgreSQL, AWS. Strong match: 6+ yrs Python, owned a payment service "
            "on EKS, deep transactional integrity, runbooks for on-call."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Backend Engineer"},
    },
    {
        "text": (
            "JD: Mid-level React Frontend Engineer for a SaaS dashboard. Strong "
            "match: 3-4 yrs React + TypeScript, shipped accessible component "
            "library, Cypress E2E, Lighthouse > 90."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Frontend Engineer"},
    },
    {
        "text": (
            "JD: Senior ML Engineer in healthcare AI. Strong match: PyTorch + "
            "MONAI experience, productionized clinical NLP model, owns model "
            "registry, SOC2 + HIPAA awareness, MICCAI paper a plus."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "ML Engineer"},
    },
    {
        "text": (
            "JD: Data Engineer at an analytics company. Strong match: 4+ yrs "
            "Spark/Airflow, dbt models for Snowflake, partition tuning at TB "
            "scale, owned a streaming pipeline (Kafka → Iceberg)."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Data Engineer"},
    },
    {
        "text": (
            "JD: DevOps Engineer for a Kubernetes platform team. Strong match: "
            "Terraform + Helm + ArgoCD, wrote a custom Kubernetes operator, "
            "GitOps experience, AWS solutions architect cert."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "DevOps Engineer"},
    },
    {
        "text": (
            "JD: Junior Data Scientist at a retail e-commerce startup. Strong "
            "match: 0-2 yrs, Kaggle medals, built recsys prototypes, comfortable "
            "with pandas + scikit-learn, conveys results clearly."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Data Scientist"},
    },
    {
        "text": (
            "JD: iOS Engineer for a fitness app. Strong match: 3+ yrs Swift + "
            "SwiftUI, HealthKit integration, App Store releases owned, Combine "
            "and async/await fluency."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Mobile iOS Developer"},
    },
    {
        "text": (
            "JD: Site Reliability Engineer at a high-traffic ad-tech firm. Strong "
            "match: 5+ yrs, SLO engineering, owned Prometheus + Grafana stack, "
            "wrote postmortems, Go for control-plane tooling."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Site Reliability Engineer"},
    },
    {
        "text": (
            "JD: LLM Engineer for an AI productivity startup. Strong match: "
            "RAG systems in production (LangChain or LlamaIndex), familiarity "
            "with Qdrant/Weaviate, evals harness, prompt versioning."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "LLM Engineer"},
    },
    {
        "text": (
            "JD: Security Engineer at a payments platform. Strong match: 4+ yrs "
            "appsec, SAST/DAST tooling, threat modeling, OWASP Top 10 fluency, "
            "OSCP a plus, hands-on incident response."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Security Engineer"},
    },
    {
        "text": (
            "JD: Senior Full-Stack Engineer at an early-stage SaaS startup. "
            "Strong match: ships features end-to-end (React + Node + Postgres), "
            "comfortable owning CI, infra-as-code, has 0→1 startup history."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Full-Stack Engineer"},
    },
    {
        "text": (
            "JD: NLP Engineer for a legal-tech firm. Strong match: 3+ yrs, "
            "transformer fine-tuning, span extraction / NER, comfortable with "
            "Hugging Face Trainer, evaluation against domain-specific test sets."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "NLP Engineer"},
    },
    {
        "text": (
            "JD: Computer Vision Engineer for an autonomous-vehicle perception "
            "team. Strong match: PyTorch + ROS2, multi-sensor fusion (LiDAR + "
            "camera), CUDA optimization, KITTI/nuScenes experience."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Computer Vision Engineer"},
    },
    {
        "text": (
            "JD: MLOps Engineer at a series-B AI company. Strong match: "
            "Kubeflow or Metaflow in production, MLflow + model registry, "
            "feature stores, drift monitoring with Evidently or Great Expectations."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "MLOps Engineer"},
    },
    {
        "text": (
            "JD: Cloud Architect for a regulated industry migration. Strong "
            "match: AWS or Azure pro-level certs, designed multi-account "
            "landing zones, network segmentation, FinOps cost optimization."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Cloud Architect"},
    },
    {
        "text": (
            "JD: Test Automation Engineer for a fintech. Strong match: 4+ yrs "
            "Selenium/Playwright + pytest/Jest, contract testing with Pact, "
            "owned a test pyramid, integrated tests into CI."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Test Automation Engineer"},
    },
    {
        "text": (
            "JD: Engineering Manager (5 IC team). Strong match: 7+ yrs IC + 2+ "
            "yrs management, ran weekly 1:1s, drove perf calibration, technical "
            "depth to unblock teams without micromanaging."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Engineering Manager"},
    },
    {
        "text": (
            "JD: Embedded Systems Engineer for IoT consumer hardware. Strong "
            "match: C/C++ on ARM Cortex-M, RTOS familiarity, BLE stack, "
            "low-power design, JTAG debugging."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Embedded Systems Engineer"},
    },
    {
        "text": (
            "JD: Senior Database Administrator. Strong match: 7+ yrs Postgres + "
            "MySQL, replication topologies, query plan tuning, backup/restore "
            "runbooks, on-call experience for tier-1 systems."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Database Administrator"},
    },
    {
        "text": (
            "JD: Developer Relations Engineer for a developer-tools startup. "
            "Strong match: prior IC eng background, conference talks, "
            "open-source contributions, technical writing portfolio."
        ),
        "metadata": {"type": "jd_resume_pair", "role": "Developer Relations Engineer"},
    },
]


def _flatten_taxonomy(taxonomy: dict[str, Any]) -> list[dict[str, Any]]:
    """Flatten the nested taxonomy into a list of self-contained text chunks."""
    docs: list[dict[str, Any]] = []
    for role, info in taxonomy.items():
        # Role overview
        overview = (
            f"Role: {role}\n"
            f"Description: {info.get('description', '')}\n"
            f"Core skills: {', '.join(info.get('core_skills', []))}\n"
            f"Tech stacks: "
            f"{', '.join(s.get('name', '') for s in info.get('tech_stacks', []))}"
        )
        docs.append(
            {"text": overview, "metadata": {"role": role, "type": "role_overview"}}
        )

        # Per-tech-stack
        for stack in info.get("tech_stacks", []):
            stack_text = (
                f"Role: {role}\n"
                f"Stack: {stack.get('name', '')}\n"
                f"Skills: {', '.join(stack.get('skills', []))}"
            )
            docs.append(
                {
                    "text": stack_text,
                    "metadata": {
                        "role": role,
                        "type": "tech_stack",
                        "stack": stack.get("name", ""),
                    },
                }
            )

        # Seniority benchmarks
        for level, desc in info.get("seniority", {}).items():
            sen_text = f"Role: {role}\nSeniority: {level}\nProfile: {desc}"
            docs.append(
                {
                    "text": sen_text,
                    "metadata": {
                        "role": role,
                        "type": "seniority_benchmark",
                        "level": level,
                    },
                }
            )

        # Common gaps
        for gap in info.get("common_gaps", []):
            gap_text = f"Role: {role}\nCommon gap: {gap}"
            docs.append(
                {
                    "text": gap_text,
                    "metadata": {"role": role, "type": "common_gap"},
                }
            )

    return docs


def main(recreate: bool = False) -> None:
    """End-to-end ingestion: load taxonomy → flatten → embed → upsert."""
    logger.info("ingest_start", taxonomy=str(TAXONOMY_PATH))

    if not TAXONOMY_PATH.exists():
        raise FileNotFoundError(f"Skill taxonomy not found: {TAXONOMY_PATH}")

    with TAXONOMY_PATH.open(encoding="utf-8") as f:
        taxonomy = json.load(f)

    if not taxonomy:
        logger.warning("taxonomy_empty", path=str(TAXONOMY_PATH))
        return

    docs: list[dict[str, Any]] = _flatten_taxonomy(taxonomy)
    docs.extend(SAMPLE_JD_RESUME_PAIRS)
    logger.info(
        "documents_prepared",
        total=len(docs),
        from_taxonomy=len(docs) - len(SAMPLE_JD_RESUME_PAIRS),
        sample_pairs=len(SAMPLE_JD_RESUME_PAIRS),
    )

    texts: list[str] = [d["text"] for d in docs]
    logger.info("embedding_documents", count=len(texts))
    vectors = embed_documents(texts)
    for d, vec in zip(docs, vectors, strict=True):
        d["vector"] = vec

    store = QdrantStore()
    store.init_collection(recreate=recreate)
    n = store.upsert(docs)

    logger.info("ingest_complete", n_documents=n, collection=store.collection)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge base into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate the collection before upserting (data lost).",
    )
    args = parser.parse_args()
    main(recreate=args.recreate)

"""Ingest the skill taxonomy and sample JD/resume pairs into Qdrant.

The taxonomy (``data/skill_taxonomy.json``) is flattened into per-role chunks and
the curated match descriptions (``data/sample_pairs.json``) are appended, giving
the RAG agent grounded examples to retrieve. Idempotent; pass ``--recreate`` to
drop and rebuild the collection.
"""

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
SAMPLE_PAIRS_PATH: Path = DATA_DIR / "sample_pairs.json"


def _load_sample_pairs() -> list[dict[str, Any]]:
    """Load the curated JD/ideal-resume match descriptions that ground retrieval."""
    if not SAMPLE_PAIRS_PATH.exists():
        logger.warning("sample_pairs_missing", path=str(SAMPLE_PAIRS_PATH))
        return []
    with SAMPLE_PAIRS_PATH.open(encoding="utf-8") as f:
        return json.load(f)


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
        docs.append({"text": overview, "metadata": {"role": role, "type": "role_overview"}})

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
    sample_pairs = _load_sample_pairs()
    docs.extend(sample_pairs)
    logger.info(
        "documents_prepared",
        total=len(docs),
        from_taxonomy=len(docs) - len(sample_pairs),
        sample_pairs=len(sample_pairs),
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

"""Qdrant client wrapper for the RecruitSense knowledge collection.

Exposes a thin ``QdrantStore`` class around qdrant-client 1.12 with:
- :meth:`init_collection` — idempotent create (or recreate) with cosine + 1024-dim vectors.
- :meth:`upsert` — batch insert/update of documents.
- :meth:`search` — kNN cosine retrieval by query vector.
- :meth:`health` — liveness check used by ``GET /health``.

A module-level singleton store is exposed via :func:`get_store` and reuses
``Settings`` for host/port/collection.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.http import models as rest

from backend.core.config import get_settings
from backend.rag.embedder import EMBEDDING_DIM
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single search result: text payload + cosine score + metadata."""

    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class QdrantStore:
    """Thin wrapper around qdrant-client for one collection."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        collection: str | None = None,
    ) -> None:
        settings = get_settings()
        self.host: str = host or settings.qdrant_host
        self.port: int = port or settings.qdrant_port
        self.collection: str = collection or settings.qdrant_collection
        self.client: QdrantClient = QdrantClient(host=self.host, port=self.port)

    def init_collection(self, recreate: bool = False) -> None:
        """Create the collection if it doesn't exist (idempotent).

        If ``recreate=True``, the existing collection is dropped first
        (data lost — use only for clean re-ingestion).
        """
        existing = {c.name for c in self.client.get_collections().collections}
        if self.collection in existing and not recreate:
            logger.info("collection_exists", name=self.collection)
            return
        if self.collection in existing and recreate:
            logger.warning("collection_recreating", name=self.collection)
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=rest.VectorParams(
                size=EMBEDDING_DIM, distance=rest.Distance.COSINE
            ),
        )
        logger.info("collection_created", name=self.collection, dim=EMBEDDING_DIM)

    def upsert(self, docs: list[dict[str, Any]]) -> int:
        """Upsert a batch of documents.

        Each ``doc`` must have keys ``text`` (str) and ``vector`` (list[float]).
        Optional: ``id`` (str/int — auto-generated if missing) and
        ``metadata`` (dict, merged into the payload alongside ``text``).

        Returns the number of points written.
        """
        if not docs:
            return 0
        points: list[rest.PointStruct] = []
        for d in docs:
            point_id = d.get("id") or str(uuid.uuid4())
            payload = {"text": d["text"], **d.get("metadata", {})}
            points.append(
                rest.PointStruct(id=point_id, vector=d["vector"], payload=payload)
            )
        self.client.upsert(collection_name=self.collection, points=points)
        logger.info("upserted", count=len(points), collection=self.collection)
        return len(points)

    def search(self, query_vector: list[float], k: int = 5) -> list[RetrievedChunk]:
        """kNN cosine search. Returns top-``k`` chunks with scores and metadata."""
        response = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=k,
        )
        chunks: list[RetrievedChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            text = payload.pop("text", "")
            chunks.append(
                RetrievedChunk(text=text, score=float(point.score), metadata=payload)
            )
        return chunks

    def health(self) -> bool:
        """Return ``True`` if Qdrant is reachable."""
        try:
            self.client.get_collections()
            return True
        except Exception as exc:
            logger.warning("qdrant_unhealthy", reason=str(exc))
            return False


_default_store: QdrantStore | None = None


def get_store() -> QdrantStore:
    """Return a process-wide singleton ``QdrantStore`` configured from env."""
    global _default_store
    if _default_store is None:
        _default_store = QdrantStore()
    return _default_store

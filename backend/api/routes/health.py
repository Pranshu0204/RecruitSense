"""``GET /health`` — liveness check + Qdrant/Redis dependency health."""

import asyncio

from fastapi import APIRouter

from backend.rag.vector_store import get_store
from backend.utils.redis_cache import is_redis_available

router: APIRouter = APIRouter()


@router.get("/health")
async def health() -> dict[str, object]:
    """Return service status and the live status of Qdrant and Redis.

    ``status`` is ``"ok"`` only if both dependencies respond; otherwise
    ``"degraded"``. Always returns 200 so external monitors can distinguish
    *up-but-degraded* from *down*.
    """
    qdrant_ok: bool = await asyncio.to_thread(lambda: get_store().health())
    redis_ok: bool = is_redis_available()
    return {
        "status": "ok" if (qdrant_ok and redis_ok) else "degraded",
        "qdrant": qdrant_ok,
        "redis": redis_ok,
    }

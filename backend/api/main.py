"""FastAPI application entrypoint for RecruitSense.

Wires the three route modules (``health``, ``screen``, ``batch``), CORS, a
structured-logging HTTP middleware, and a lifespan handler that logs the
configured model + dependency hosts at startup.

Run with::

    make run                           # uvicorn backend.api.main:app --reload --port 8000
    uvicorn backend.api.main:app       # production
"""

import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.api.routes import batch, health, screen, session
from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook — logs configuration and is a place to warm caches."""
    settings = get_settings()
    logger.info(
        "startup",
        model=settings.default_model,
        qdrant=f"{settings.qdrant_host}:{settings.qdrant_port}",
        redis=f"{settings.redis_host}:{settings.redis_port}",
        log_level=settings.log_level,
    )
    yield
    logger.info("shutdown")


app: FastAPI = FastAPI(
    title="RecruitSense API",
    description=(
        "LLM-powered resume screening with RAG (Qdrant + BGE-large), "
        "LangGraph multi-agent orchestration, and bias detection."
    ),
    version="0.1.0",
    lifespan=lifespan,
)
# --- CORS ---------------------------------------------------------------------
# `*` is fine for local Streamlit dev; tighten the allowlist for production.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Structured request/response logging --------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Emit one structured log line per HTTP request with latency in ms."""
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception as exc:
        duration_ms = (time.perf_counter() - start) * 1000
        logger.error(
            "http_request_unhandled",
            method=request.method,
            path=request.url.path,
            duration_ms=round(duration_ms, 1),
            error=str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})
    duration_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "http_request",
        method=request.method,
        path=request.url.path,
        status=response.status_code,
        duration_ms=round(duration_ms, 1),
    )
    return response


# --- Routers ------------------------------------------------------------------

app.include_router(health.router, tags=["health"])
app.include_router(screen.router, tags=["screening"])
app.include_router(batch.router, tags=["screening"])
app.include_router(session.router)


@app.get("/", include_in_schema=False)
async def root() -> dict[str, str]:
    """Lightweight landing endpoint with pointer to docs."""
    return {"name": "RecruitSense API", "version": "0.1.0", "docs": "/docs"}

"""Redis-backed cache decorator for LLM calls.

Designed to gracefully degrade — if Redis is unreachable, the wrapped function
runs as if uncached and a warning is logged once. Cache keys include
``PROMPT_VERSION`` so prompt edits invalidate the cache automatically.

The wrapped function MUST return a JSON-serializable value (``dict``, ``list``,
primitive, or anything ``json.dumps`` can handle with ``default=str``).
Pydantic models should be ``.model_dump()``-ed by the wrapped function before
returning so that cache hits and misses both yield the same shape.

The decorator transparently handles both sync and async functions::

    @cache_llm(namespace="scorer")
    async def score(jd: str, resume: str) -> dict: ...

    @cache_llm(namespace="parser", ttl=600)
    def parse(text: str) -> dict: ...
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import inspect
import json
from collections.abc import Callable
from typing import Any, TypeVar

import redis

from backend.core.config import get_settings
from backend.core.prompts import PROMPT_VERSION
from backend.utils.logger import get_logger

logger = get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Any])

_client: redis.Redis | None = None
_redis_warned: bool = False


def _get_client() -> redis.Redis | None:
    """Lazy-init a Redis client. Returns ``None`` if Redis is unreachable."""
    global _client, _redis_warned
    if _client is not None:
        return _client

    settings = get_settings()
    try:
        client = redis.Redis(
            host=settings.redis_host,
            port=settings.redis_port,
            decode_responses=True,
            socket_connect_timeout=1.0,
        )
        client.ping()
        _client = client
        return _client
    except Exception as exc:
        if not _redis_warned:
            logger.warning("redis_unavailable", reason=str(exc))
            _redis_warned = True
        return None


def is_redis_available() -> bool:
    """Public health-check helper: ``True`` iff Redis ping succeeds."""
    return _get_client() is not None


def _make_key(namespace: str, args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Build a deterministic cache key: prompt-version + namespace + sha256(args)."""
    payload = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"recruitsense:{PROMPT_VERSION}:{namespace}:{digest}"


def cache_llm(namespace: str, ttl: int | None = None) -> Callable[[F], F]:
    """Cache the JSON-serializable return value of a function in Redis.

    Args:
        namespace: short tag included in the cache key (``"parser"``, ``"scorer"`` …).
        ttl: TTL in seconds; defaults to ``Settings.redis_ttl_seconds``.

    Returns:
        A decorator that wraps either sync or async functions.
    """

    def decorator(func: F) -> F:
        is_async = inspect.iscoroutinefunction(func)

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = _get_client()
            ttl_s = ttl if ttl is not None else get_settings().redis_ttl_seconds
            key = _make_key(namespace, args, kwargs)

            if client is not None:
                try:
                    cached = client.get(key)
                    if cached is not None:
                        logger.debug("cache_hit", namespace=namespace)
                        return json.loads(cached)
                except Exception as exc:
                    logger.warning("cache_get_failed", reason=str(exc))

            result = func(*args, **kwargs)

            if client is not None:
                try:
                    client.setex(key, ttl_s, json.dumps(result, default=str))
                except Exception as exc:
                    logger.warning("cache_set_failed", reason=str(exc))
            return result

        @functools.wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            client = _get_client()
            ttl_s = ttl if ttl is not None else get_settings().redis_ttl_seconds
            key = _make_key(namespace, args, kwargs)

            if client is not None:
                try:
                    cached = await asyncio.to_thread(client.get, key)
                    if cached is not None:
                        logger.debug("cache_hit", namespace=namespace)
                        return json.loads(cached)
                except Exception as exc:
                    logger.warning("cache_get_failed", reason=str(exc))

            result = await func(*args, **kwargs)

            if client is not None:
                try:
                    await asyncio.to_thread(
                        client.setex, key, ttl_s, json.dumps(result, default=str)
                    )
                except Exception as exc:
                    logger.warning("cache_set_failed", reason=str(exc))
            return result

        return async_wrapper if is_async else sync_wrapper  # type: ignore[return-value]

    return decorator

"""Redis-cache decorator tests — focused on graceful degradation when Redis is down."""

from __future__ import annotations

import asyncio

import pytest

from backend.utils import redis_cache
from backend.utils.redis_cache import _make_key, cache_llm


def test_make_key_is_deterministic() -> None:
    """Same args + kwargs → same key (and includes namespace + prompt-version)."""
    k1 = _make_key("ns", ("a", 1), {"k": "v"})
    k2 = _make_key("ns", ("a", 1), {"k": "v"})
    assert k1 == k2
    assert k1.startswith("recruitsense:")
    assert ":ns:" in k1


def test_make_key_differs_when_args_change() -> None:
    assert _make_key("ns", ("a",), {}) != _make_key("ns", ("b",), {})


def test_make_key_differs_when_namespace_changes() -> None:
    assert _make_key("a", (), {}) != _make_key("b", (), {})


def test_sync_decorator_runs_function_when_redis_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """When Redis is unreachable, the wrapped function must still execute."""
    monkeypatch.setattr(redis_cache, "_get_client", lambda: None)
    calls = {"n": 0}

    @cache_llm(namespace="test_sync")
    def add(a: int, b: int) -> int:
        calls["n"] += 1
        return a + b

    assert add(2, 3) == 5
    assert add(2, 3) == 5  # no caching → function runs again
    assert calls["n"] == 2


def test_async_decorator_runs_function_when_redis_down(monkeypatch: pytest.MonkeyPatch) -> None:
    """Same as above but for ``async def`` functions."""
    monkeypatch.setattr(redis_cache, "_get_client", lambda: None)
    calls = {"n": 0}

    @cache_llm(namespace="test_async")
    async def add(a: int, b: int) -> int:
        calls["n"] += 1
        await asyncio.sleep(0)
        return a + b

    result = asyncio.run(add(1, 2))
    assert result == 3
    assert calls["n"] == 1


def test_sync_decorator_returns_cached_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """When the fake client returns a cached value, the function must NOT execute."""
    import json

    class FakeClient:
        def __init__(self) -> None:
            self.store: dict[str, str] = {}

        def get(self, key: str) -> str | None:
            return self.store.get(key)

        def setex(self, key: str, _ttl: int, value: str) -> None:
            self.store[key] = value

    fake = FakeClient()
    monkeypatch.setattr(redis_cache, "_get_client", lambda: fake)

    calls = {"n": 0}

    @cache_llm(namespace="cached")
    def expensive(x: int) -> dict[str, int]:
        calls["n"] += 1
        return {"value": x * x}

    assert expensive(4) == {"value": 16}
    assert calls["n"] == 1
    # second call should hit the cache
    assert expensive(4) == {"value": 16}
    assert calls["n"] == 1
    # different arg → cache miss → function runs again
    assert expensive(5) == {"value": 25}
    assert calls["n"] == 2

    # Sanity: cache actually contains JSON
    cached_keys = list(fake.store.keys())
    assert cached_keys
    assert json.loads(fake.store[cached_keys[0]])["value"] in (16, 25)

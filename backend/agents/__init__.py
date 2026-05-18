

from __future__ import annotations

from langchain_openai import ChatOpenAI

from backend.core.config import get_settings


def get_llm(model: str | None = None, temperature: float = 0.1) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance configured for OpenRouter.

    Args:
        model: model id (e.g. ``mistralai/mistral-7b-instruct``); defaults to
            ``Settings.default_model``.
        temperature: sampling temperature; low (0.1) for deterministic scoring,
            slightly higher for sub-query generation.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=model or settings.default_model,
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
    )

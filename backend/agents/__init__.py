"""RecruitSense agents package — shared LLM factory.

Exposes :func:`get_llm`, a thin wrapper that builds a ``ChatOpenAI`` client
pointed at OpenRouter (or any OpenAI-compatible endpoint via env). Each agent
module imports this factory rather than instantiating its own client, so model
selection is centralized.
"""

from langchain_openai import ChatOpenAI

from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

# Allowlisted free-tier OpenRouter models. A request for anything outside this
# set (e.g. a stale UI selection of a rate-limited or paid model) is ignored and
# the configured default is used instead, so the server can never be driven onto
# a model the account can't use.
FREE_MODELS: frozenset[str] = frozenset(
    {
        "openai/gpt-oss-120b:free",
        "qwen/qwen3-next-80b-a3b-instruct:free",
        "z-ai/glm-4.5-air:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "nvidia/nemotron-3-ultra-550b-a55b:free",
    }
)


def resolve_model(model: str | None) -> str:
    """Return a usable model id.

    Falls back to ``Settings.default_model`` when ``model`` is empty or not in
    the free-tier allowlist, so a stale or invalid client-supplied model never
    reaches the provider.
    """
    default = get_settings().default_model
    if not model or model == default:
        return default
    if model not in FREE_MODELS:
        logger.warning("model_not_allowed", requested=model, using=default)
        return default
    return model


def get_llm(model: str | None = None, temperature: float = 0.1) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance configured for OpenRouter.

    ``model`` is validated against the free-tier allowlist via
    :func:`resolve_model` before being forwarded to the client.
    """
    settings = get_settings()
    return ChatOpenAI(
        model=resolve_model(model),
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        temperature=temperature,
        max_tokens=4096,
        # Free-tier models are frequently rate-limited (429); retry with backoff
        # before giving up so a transient limit doesn't produce a fallback score.
        max_retries=4,
    )

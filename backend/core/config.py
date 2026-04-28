"""Application settings loaded from environment / .env via Pydantic v2 BaseSettings.

A single ``Settings`` instance is exposed through :func:`get_settings`, which is
``lru_cache``-wrapped so it acts as a process-wide singleton. All values can be
overridden via environment variables (case-insensitive) or a ``.env`` file at
the project root.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Centralized configuration for the RecruitSense backend."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- OpenRouter LLM ---
    openrouter_api_key: str = Field(default="", description="OpenRouter API key.")
    openrouter_base_url: str = Field(default="https://openrouter.ai/api/v1")
    default_model: str = Field(default="mistralai/mistral-7b-instruct")

    # --- Qdrant ---
    qdrant_host: str = Field(default="localhost")
    qdrant_port: int = Field(default=6333, ge=1, le=65535)
    qdrant_collection: str = Field(default="recruitsense_knowledge")

    # --- Redis ---
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379, ge=1, le=65535)
    redis_ttl_seconds: int = Field(default=3600, ge=0)

    # --- Fine-tuning ---
    finetune_base_model: str = Field(default="mistralai/Mistral-7B-Instruct-v0.3")
    mlflow_tracking_uri: str = Field(default="http://localhost:5000")

    # --- Logging ---
    log_level: str = Field(default="INFO")

    # --- API ---
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000, ge=1, le=65535)

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, v: str) -> str:
        upper = v.upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}, got '{v}'")
        return upper


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached ``Settings`` singleton."""
    return Settings()

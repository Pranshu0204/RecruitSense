"""Structured JSON logger using structlog.

Each log line is a single JSON object with timestamp, level, logger name,
event, and arbitrary key/value extras passed by callers. The configured level
comes from ``Settings.log_level``.

Usage::

    from backend.utils.logger import get_logger
    log = get_logger(__name__)
    log.info("resume_scored", candidate="Jane Doe", composite=87.4)
"""

from __future__ import annotations

import logging
import sys

import structlog

from backend.core.config import get_settings

_configured: bool = False


def _configure() -> None:
    """Apply structlog + stdlib logging configuration exactly once per process."""
    global _configured
    if _configured:
        return

    settings = get_settings()
    level = getattr(logging, settings.log_level, logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _configured = True


def get_logger(name: str = __name__) -> structlog.stdlib.BoundLogger:
    """Return a structlog logger bound with the given name."""
    _configure()
    return structlog.get_logger(name)

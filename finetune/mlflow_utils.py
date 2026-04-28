"""MLflow experiment logging helpers.

Wraps :func:`mlflow.start_run` so the rest of the fine-tuning pipeline never
imports MLflow directly. If MLflow is unreachable (e.g. no tracking server
running locally), :func:`active_run` degrades to a no-op context manager so
``train.py`` / ``evaluate.py`` keep working — useful for quick local smoke
tests where you don't want to spin up a tracking server.

The tracking URI is taken from :class:`backend.core.config.Settings`
(``MLFLOW_TRACKING_URI``, default ``http://localhost:5000``).
"""

from __future__ import annotations

import contextlib
import os
from pathlib import Path
from typing import Any, Iterator

import mlflow

from backend.core.config import get_settings
from backend.utils.logger import get_logger

logger = get_logger(__name__)

EXPERIMENT_NAME = "recruitsense-finetune"


def _configure() -> bool:
    """Point MLflow at the configured tracking URI. Returns ``False`` on failure."""
    try:
        uri = os.environ.get("MLFLOW_TRACKING_URI") or get_settings().mlflow_tracking_uri
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(EXPERIMENT_NAME)
        return True
    except Exception as exc:
        logger.warning("mlflow_configure_failed", reason=str(exc))
        return False


@contextlib.contextmanager
def active_run(run_name: str, tags: dict[str, str] | None = None) -> Iterator[Any]:
    """Context manager that yields an active MLflow run, or a no-op stand-in."""
    if not _configure():
        yield None
        return
    try:
        with mlflow.start_run(run_name=run_name, tags=tags or {}) as run:
            yield run
    except Exception as exc:
        logger.warning("mlflow_run_failed", reason=str(exc))
        yield None


def log_params(params: dict[str, Any]) -> None:
    """Log a flat dict of params, coercing values to MLflow-friendly strings."""
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_params({k: _safe(v) for k, v in params.items()})
    except Exception as exc:
        logger.warning("mlflow_log_params_failed", reason=str(exc))


def log_metrics(metrics: dict[str, float], step: int | None = None) -> None:
    """Log a flat dict of numeric metrics."""
    if mlflow.active_run() is None:
        return
    try:
        mlflow.log_metrics(
            {k: float(v) for k, v in metrics.items() if v is not None}, step=step
        )
    except Exception as exc:
        logger.warning("mlflow_log_metrics_failed", reason=str(exc))


def log_artifact(path: str | Path) -> None:
    """Log a file or directory as an MLflow artifact."""
    if mlflow.active_run() is None:
        return
    try:
        path = Path(path)
        if path.is_dir():
            mlflow.log_artifacts(str(path))
        else:
            mlflow.log_artifact(str(path))
    except Exception as exc:
        logger.warning("mlflow_log_artifact_failed", reason=str(exc), path=str(path))


def _safe(v: Any) -> str:
    """Coerce arbitrary values to short strings MLflow can store."""
    s = str(v)
    return s if len(s) < 500 else s[:497] + "..."

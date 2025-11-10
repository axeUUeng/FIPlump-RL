"""Lightweight helpers for optional MLflow logging."""

from __future__ import annotations

from numbers import Number
from typing import Any, Mapping, Optional

try:  # pragma: no cover - optional dependency
    import mlflow
except ImportError:  # pragma: no cover - gracefully degrade when MLflow is absent
    mlflow = None  # type: ignore[assignment]


def _has_active_run() -> bool:
    return mlflow is not None and mlflow.active_run() is not None


def ensure_run(run_name: str) -> bool:
    """Start an MLflow run if one is not already active."""

    if mlflow is None:
        return False
    if mlflow.active_run() is not None:
        return False
    mlflow.start_run(run_name=run_name)
    return True


def end_run_if_started(started: bool) -> None:
    """Close the run that was started by ``ensure_run``."""

    if mlflow is not None and started:
        mlflow.end_run()


def log_params(params: Mapping[str, Any]) -> None:
    if not _has_active_run():
        return
    sanitized = {str(key): _sanitize_param_value(value) for key, value in params.items()}
    if sanitized:
        mlflow.log_params(sanitized)


def log_metric(name: str, value: float, step: Optional[int] = None) -> None:
    if not _has_active_run():
        return
    mlflow.log_metric(name, float(value), step=step)


def set_tag(key: str, value: Any) -> None:
    if not _has_active_run():
        return
    mlflow.set_tag(key, value)


def _sanitize_param_value(value: Any) -> Any:
    if isinstance(value, (str, bool)):
        return value
    if isinstance(value, Number):
        return value
    return str(value)

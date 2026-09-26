"""Request-building helper shared across test modules."""
from __future__ import annotations

from app.models import BaselineSpec, EvalRequest, Sample


def make_request(
    *,
    tenant_id: str = "ten",
    project_id: str | None = None,
    model_version: str = "v1",
    samples=None,
    metrics=None,
    config=None,
    baseline=None,
    timeout_seconds=None,
    idempotency_key=None,
):
    if samples is None:
        samples = [Sample(id="s1", reference="hello", prediction="hello")]
    if metrics is None:
        metrics = ["exact_match"]
    return EvalRequest(
        tenant_id=tenant_id,
        project_id=project_id,
        model_version=model_version,
        samples=list(samples),
        metrics=list(metrics),
        config=dict(config or {}),
        baseline=BaselineSpec(**baseline) if baseline else None,
        timeout_seconds=timeout_seconds,
        idempotency_key=idempotency_key,
    )


def wait_terminal(service, task_id, timeout=10.0):
    service.executor.wait(task_id, timeout=timeout)
    return service.get_status(task_id)

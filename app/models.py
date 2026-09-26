"""Domain & API data models plus deterministic serialization helpers."""
from __future__ import annotations

import enum
import hashlib
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator


class TaskState(str, enum.Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


TERMINAL_STATES = frozenset(
    {TaskState.SUCCEEDED, TaskState.FAILED, TaskState.CANCELLED, TaskState.TIMEOUT}
)
NON_TERMINAL_STATES = frozenset({TaskState.PENDING, TaskState.RUNNING})


class Sample(BaseModel):
    id: str
    input: str = ""
    reference: str | None = None
    prediction: str | None = None


class BaselineSpec(BaseModel):
    task_id: str | None = None
    metrics: dict[str, float] | None = None


class EvalRequest(BaseModel):
    tenant_id: str
    project_id: str | None = None
    model_version: str
    samples: list[Sample]
    metrics: list[str] = Field(default_factory=lambda: ["exact_match"])
    config: dict[str, Any] = Field(default_factory=dict)
    baseline: BaselineSpec | None = None
    timeout_seconds: float | None = None
    idempotency_key: str | None = None

    @field_validator("tenant_id", "model_version")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be a non-empty string")
        return value

    @field_validator("samples")
    @classmethod
    def _unique_sample_ids(cls, samples: list[Sample]) -> list[Sample]:
        if not samples:
            raise ValueError("at least one sample is required")
        ids = [s.id for s in samples]
        if any(not sid or not sid.strip() for sid in ids):
            raise ValueError("sample ids must be non-empty")
        if len(set(ids)) != len(ids):
            duplicates = sorted({sid for sid in ids if ids.count(sid) > 1})
            raise ValueError(f"duplicate sample ids: {duplicates}")
        return samples

    @field_validator("metrics")
    @classmethod
    def _non_empty_metrics(cls, metrics: list[str]) -> list[str]:
        if not metrics:
            raise ValueError("at least one metric is required")
        if any(not name for name in metrics):
            raise ValueError("metric names must be non-empty")
        return metrics

    @field_validator("timeout_seconds")
    @classmethod
    def _positive_timeout(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError("timeout_seconds must be positive")
        return value

    @model_validator(mode="after")
    def _baseline_shape(self) -> EvalRequest:
        if (self.baseline is not None
                and self.baseline.task_id is None
                and not self.baseline.metrics):
            raise ValueError("baseline requires task_id or metrics")
        return self


class SampleScore(BaseModel):
    sample_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class MetricComparison(BaseModel):
    metric: str
    current: float
    baseline: float
    delta: float


class TaskResult(BaseModel):
    # ``None`` means the metric could not be aggregated (no successful sample).
    metrics: dict[str, float | None]
    sample_scores: list[SampleScore]
    total_samples: int
    succeeded_samples: int
    failed_samples: int
    comparison: dict[str, MetricComparison] | None = None


class TaskStatus(BaseModel):
    task_id: str
    tenant_id: str
    project_id: str | None
    model_version: str
    state: TaskState
    created_at: float
    started_at: float | None
    finished_at: float | None
    error: str | None
    result: TaskResult | None = None


class SubmitResponse(BaseModel):
    task_id: str
    state: TaskState
    reused: bool


class CancelResponse(BaseModel):
    task_id: str
    state: TaskState


class QuotaUpdate(BaseModel):
    tenant_id: str
    project_id: str | None = None
    limit: int = Field(ge=0)


class QuotaInfo(BaseModel):
    scope: str
    tenant_id: str
    project_id: str | None
    limit: int
    used: int


def quota_scope(tenant_id: str, project_id: str | None = None) -> str:
    """Canonical quota/ownership scope identifier."""
    if project_id:
        return f"{tenant_id}:{project_id}"
    return tenant_id


def canonical_json(obj: Any) -> str:
    """Deterministic JSON serialization.

    Dictionary keys are sorted and compact separators are used so that the
    same logical value always yields the same byte string. Floats go through
    ``repr`` (Python's shortest round-trip representation) which is itself
    deterministic.
    """
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def _fingerprint_payload(req: EvalRequest) -> dict[str, Any]:
    baseline: Any = None
    if req.baseline is not None:
        baseline = {
            "task_id": req.baseline.task_id,
            "metrics": req.baseline.metrics,
        }
    return {
        # Ownership is part of the key: identical workloads for different
        # tenants/projects are independent tasks.
        "tenant_id": req.tenant_id,
        "project_id": req.project_id,
        "model_version": req.model_version,
        "samples": [s.model_dump() for s in req.samples],
        # Order of requested metrics does not change the evaluation.
        "metrics": sorted(req.metrics),
        "config": req.config,
        "baseline": baseline,
        "idempotency_key": req.idempotency_key,
        # NOTE: timeout_seconds intentionally excluded: execution parameter,
        # not part of the evaluation specification.
    }


def request_fingerprint(req: EvalRequest) -> str:
    """Stable SHA-256 fingerprint of the evaluation specification.

    Two requests with the same fingerprint reuse the same task; any change in
    model version, samples, metrics/config, baseline, tenant/project scope or
    idempotency key produces a different fingerprint (cache invalidation).
    """
    raw = canonical_json(_fingerprint_payload(req))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

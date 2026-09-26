"""Type definitions for the evaluation service (skeleton)."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class TaskStatus(str, Enum):
    """Lifecycle states of an evaluation task."""

    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


TERMINAL_STATES: frozenset[TaskStatus] = frozenset(
    {TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT}
)


class Sample(BaseModel):
    id: str
    input: str
    expected: str | None = None


class MetricSpec(BaseModel):
    name: str
    params: dict[str, Any] = Field(default_factory=dict)


class EvalRequest(BaseModel):
    tenant: str = "default"
    project: str | None = None
    model_version: str
    samples: list[Sample]
    metrics: list[MetricSpec] = Field(
        default_factory=lambda: [MetricSpec(name="exact_match")]
    )
    baseline: dict[str, float] | None = None
    # Execution parameters below do NOT affect results and are excluded
    # from the request fingerprint.
    timeout_seconds: float | None = None
    sample_delay_seconds: float = 0.0


class SampleScore(BaseModel):
    sample_id: str
    scores: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class EvalResult(BaseModel):
    """Pure function of (request, fingerprint): fully reproducible."""

    model_version: str
    fingerprint: str
    sample_scores: list[SampleScore] = Field(default_factory=list)
    aggregate: dict[str, float] = Field(default_factory=dict)
    failed_samples: list[str] = Field(default_factory=list)
    failure_reasons: dict[str, str] = Field(default_factory=dict)
    baseline: dict[str, float] | None = None
    delta_vs_baseline: dict[str, float] | None = None


class TaskRecord(BaseModel):
    task_id: str
    tenant: str
    project: str | None = None
    fingerprint: str
    status: TaskStatus
    request: EvalRequest
    result: EvalResult | None = None
    error: str | None = None
    created_at: float = 0.0
    updated_at: float = 0.0

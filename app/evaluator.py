"""Deterministic evaluation core.

A :class:`ModelBackend` turns ``(model_version, text, config)`` into a
prediction. The default backend is deterministic (same version + input always
yields the same output) and can optionally simulate latency, which makes the
whole service reproducible and testable without an external model. Samples
that carry their own ``prediction`` skip the backend entirely.
"""
from __future__ import annotations

import hashlib
import threading
import time
from typing import Protocol

from app.metrics import compute_metric
from app.models import EvalRequest, MetricComparison, SampleScore, TaskResult


class TaskCancelled(Exception):
    """Raised when an evaluation observes an explicit cancellation."""


class TaskTimeout(Exception):
    """Raised when an evaluation exceeds its deadline."""


class CancellationToken:
    """Cooperative cancellation/timeout flag polled between samples.

    The first trigger wins: an explicit cancel and a watchdog timeout racing
    each other produce a single, deterministic outcome.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None

    def _trigger(self, reason: str) -> None:
        with self._lock:
            if self._reason is None:
                self._reason = reason
                self._event.set()

    def cancel(self) -> None:
        self._trigger("cancelled")

    def request_timeout(self) -> None:
        self._trigger("timeout")

    @property
    def reason(self) -> str | None:
        return self._reason

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            if self._reason == "timeout":
                raise TaskTimeout()
            raise TaskCancelled()

    def wait(self, seconds: float) -> bool:
        """Wait for a latency window; returns True if triggered in the wait."""
        return self._event.wait(seconds)


class ModelBackend(Protocol):
    def predict(
        self, model_version: str, text: str, config: dict
    ) -> str: ...


class DeterministicBackend:
    """Local, dependency-free, reproducible 'model'.

    The prediction is derived solely from the model version, input text and a
    small piece of config, via SHA-256 — therefore identical across runs and
    processes. ``latency`` config makes a sample sleep (in interruptible
    increments) for cancellation/timeout tests without affecting output.
    """

    def predict(
        self, model_version: str, text: str, config: dict
    ) -> str:
        seed = config.get("backend_seed", "")
        digest = hashlib.sha256(
            f"{model_version}\x00{seed}\x00{text}".encode()
        ).hexdigest()
        return f"{model_version}:{digest[:12]}"


#: Backends keyed by name; ``default`` is the deterministic local backend.
_BACKENDS: dict[str, ModelBackend] = {"default": DeterministicBackend()}


def register_backend(name: str, backend: ModelBackend) -> None:
    _BACKENDS[name] = backend


def get_backend(name: str = "default") -> ModelBackend:
    return _BACKENDS[name]


def _latency(config: dict) -> float:
    value = config.get("latency", 0.0)
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _sleep_interruptible(seconds: float, token: CancellationToken,
                         deadline: float | None) -> None:
    """Sleep in small slices so cancellation/timeout are responsive."""
    remaining = seconds
    slice_len = 0.02
    while remaining > 0:
        token.raise_if_cancelled()
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskTimeout()
        if token.wait(min(slice_len, remaining)):
            token.raise_if_cancelled()
        remaining -= slice_len


def evaluate(
    req: EvalRequest,
    token: CancellationToken,
    *,
    backend: ModelBackend | None = None,
    deadline: float | None = None,
    baseline_metrics: dict[str, float] | None = None,
) -> TaskResult:
    """Run all samples deterministically.

    Raises :class:`TaskCancelled` or :class:`TaskTimeout` between samples (or
    during simulated latency); those produce no result, so no partial output
    is ever persisted. Per-sample metric/prediction errors are captured on the
    sample score and reflected in the summary instead of aborting the run.
    """
    backend = backend or get_backend(req.config.get("backend", "default"))
    sample_scores: list[SampleScore] = []
    # metric name -> list of per-sample values from successful samples only
    accum: dict[str, list[float]] = {name: [] for name in req.metrics}
    succeeded = 0

    for sample in req.samples:
        token.raise_if_cancelled()
        if deadline is not None and time.monotonic() >= deadline:
            raise TaskTimeout()

        latency = _latency(req.config)
        if latency:
            _sleep_interruptible(latency, token, deadline)

        scores: dict[str, float] = {}
        error: str | None = None
        try:
            if sample.prediction is not None:
                prediction = sample.prediction
            else:
                prediction = backend.predict(
                    req.model_version, sample.input, req.config
                )
            if sample.reference is None:
                raise ValueError("sample has no reference; cannot score")
            for name in req.metrics:
                scores[name] = compute_metric(
                    name, prediction, sample.reference, req.config
                )
        except TaskCancelled:
            raise
        except Exception as exc:  # captured per-sample, never silent
            error = f"{type(exc).__name__}: {exc}"

        sample_scores.append(
            SampleScore(sample_id=sample.id, scores=scores, error=error)
        )
        if error is None:
            succeeded += 1
            for name, value in scores.items():
                accum[name].append(value)

    token.raise_if_cancelled()

    metrics: dict[str, float | None] = {}
    for name in req.metrics:
        values = accum[name]
        metrics[name] = (sum(values) / len(values)) if values else None

    comparison: dict[str, MetricComparison] | None = None
    if baseline_metrics is not None:
        comparison = {}
        for name in req.metrics:
            base = baseline_metrics.get(name)
            current = metrics[name]
            if base is None or current is None:
                continue
            delta = current - base
            comparison[name] = MetricComparison(
                metric=name, current=current, baseline=base, delta=delta
            )
        comparison = comparison or None

    return TaskResult(
        metrics=metrics,
        sample_scores=sample_scores,
        total_samples=len(req.samples),
        succeeded_samples=succeeded,
        failed_samples=len(req.samples) - succeeded,
        comparison=comparison,
    )

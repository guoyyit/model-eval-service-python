"""Deterministic evaluation execution engine.

The runner is a pure function of the request: given the same request it
always produces the same EvalResult. Cancellation and timeout are
signalled by raising EvaluationCancelled / EvaluationTimeout, so no
partial result object is ever produced -- the caller simply persists
nothing in those cases.
"""

from __future__ import annotations

import threading
import time

from .metrics import MetricError, get_metric, simulate_model_output
from .models import EvalRequest, EvalResult, SampleScore


class EvaluationCancelled(Exception):
    """Raised when a task is cancelled mid-flight."""


class EvaluationTimeout(Exception):
    """Raised when a task exceeds its deadline."""


def _check_cancel(cancel_event: threading.Event) -> None:
    if cancel_event.is_set():
        raise EvaluationCancelled("cancelled by user")


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise EvaluationTimeout("evaluation exceeded its timeout")


def _interruptible_sleep(
    delay: float, cancel_event: threading.Event, deadline: float | None
) -> None:
    """Sleep in small slices so cancel/timeout are honoured promptly."""
    end = time.monotonic() + delay
    while True:
        _check_cancel(cancel_event)
        _check_deadline(deadline)
        remaining = end - time.monotonic()
        if remaining <= 0:
            return
        cancel_event.wait(min(remaining, 0.005))


def run_evaluation(
    request: EvalRequest,
    fingerprint: str,
    cancel_event: threading.Event,
    deadline: float | None,
) -> EvalResult:
    """Execute an evaluation deterministically.

    Failed samples are recorded with their reason and excluded from the
    aggregates -- they are never silently counted as successes.
    """
    sample_scores: list[SampleScore] = []
    failed_samples: list[str] = []
    failure_reasons: dict[str, str] = {}
    totals = {m.name: 0.0 for m in request.metrics}
    counts = {m.name: 0 for m in request.metrics}

    for sample in request.samples:
        _check_cancel(cancel_event)
        _check_deadline(deadline)
        if request.sample_delay_seconds > 0:
            _interruptible_sleep(request.sample_delay_seconds, cancel_event, deadline)

        output = simulate_model_output(request.model_version, sample.input)
        scores: dict[str, float] = {}
        error: str | None = None
        try:
            for spec in request.metrics:
                scores[spec.name] = get_metric(spec.name)(
                    output, sample.expected, spec.params
                )
        except MetricError as exc:
            scores = {}
            error = str(exc)

        if error is not None:
            failed_samples.append(sample.id)
            failure_reasons[sample.id] = error
        else:
            for name, value in scores.items():
                totals[name] += value
                counts[name] += 1
        sample_scores.append(
            SampleScore(sample_id=sample.id, scores=scores, error=error)
        )

    aggregate = {
        name: (totals[name] / counts[name] if counts[name] else 0.0) for name in totals
    }
    delta: dict[str, float] | None = None
    if request.baseline is not None:
        delta = {
            name: aggregate.get(name, 0.0) - base
            for name, base in request.baseline.items()
        }

    return EvalResult(
        model_version=request.model_version,
        fingerprint=fingerprint,
        sample_scores=sample_scores,
        aggregate=aggregate,
        failed_samples=failed_samples,
        failure_reasons=failure_reasons,
        baseline=request.baseline,
        delta_vs_baseline=delta,
    )

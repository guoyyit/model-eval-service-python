"""Requirements 3/6/7/8: cancellation, timeout, no partial results,
explainable states, and sample-level failure recording."""
from __future__ import annotations

import time

from helpers import make_request

from app.models import Sample, TaskState


def _slow_request(tenant="cancel", n=30, latency=0.05, timeout=None,
                  model_version="v1", idempotency_key=None):
    return make_request(
        tenant_id=tenant,
        model_version=model_version,
        samples=[Sample(id=str(i), reference="r") for i in range(n)],
        config={"latency": latency},
        timeout_seconds=timeout,
        idempotency_key=idempotency_key,
    )


def test_cancel_running_task_leaves_no_partial_result(service):
    r = service.submit(_slow_request())
    time.sleep(0.12)
    assert service.get_status(r.task_id).state is TaskState.RUNNING
    cancelled = service.cancel(r.task_id)
    # The returned state is explainable: running while aborting, or cancelled.
    assert cancelled.state in (TaskState.RUNNING, TaskState.CANCELLED)
    service.executor.wait(r.task_id, timeout=10)
    final = service.get_status(r.task_id)
    assert final.state is TaskState.CANCELLED
    assert final.result is None
    assert final.error == "cancelled by request"


def test_cancel_queued_task_is_immediate(service):
    # Occupy all 4 workers with long tasks.
    occupying = []
    for i in range(4):
        rr = service.submit(_slow_request(
            tenant="queue", n=100, latency=5.0, idempotency_key=f"occ{i}"))
        occupying.append(rr.task_id)
    time.sleep(0.1)
    queued = service.submit(_slow_request(
        tenant="queue", n=1, latency=0.01, idempotency_key="late"))
    assert service.get_status(queued.task_id).state is TaskState.PENDING
    out = service.cancel(queued.task_id)
    assert out.state is TaskState.CANCELLED
    assert service.get_status(queued.task_id).result is None
    # cleanup workers
    for tid in occupying:
        service.cancel(tid)
    service.executor.wait(occupying[-1], timeout=5)


def test_cancel_terminal_task_is_idempotent_noop(service):
    r = service.submit(make_request(idempotency_key="done"))
    service.executor.wait(r.task_id, timeout=10)
    out1 = service.cancel(r.task_id)
    out2 = service.cancel(r.task_id)
    assert out1.state is TaskState.SUCCEEDED
    assert out2.state is TaskState.SUCCEEDED
    assert service.get_result(r.task_id) is not None


def test_timeout_marks_timeout_without_result(service):
    r = service.submit(_slow_request(
        tenant="timeout", n=10, latency=0.1, timeout=0.15))
    service.executor.wait(r.task_id, timeout=10)
    final = service.get_status(r.task_id)
    assert final.state is TaskState.TIMEOUT
    assert final.result is None
    assert final.error == "evaluation timed out"


def test_completed_result_beats_late_cancel(service):
    r = service.submit(make_request(config={"latency": 0.05}))
    service.executor.wait(r.task_id, timeout=10)
    out = service.cancel(r.task_id)
    assert out.state is TaskState.SUCCEEDED
    assert out.result is not None


def test_failed_samples_are_recorded_not_counted_as_success(service):
    req = make_request(
        tenant_id="fails",
        samples=[
            Sample(id="ok", reference="1.5", prediction="1.5"),
            Sample(id="no-ref", reference=None, prediction="p"),
            Sample(id="bad-number", reference="1.0", prediction="oops"),
        ],
        metrics=["exact_match", "numeric_abs_error"],
    )
    r = service.submit(req)
    service.executor.wait(r.task_id, timeout=10)
    result = service.get_result(r.task_id)
    # Task completes; failures are explicit, not silent.
    assert result.total_samples == 3
    assert result.failed_samples == 2
    assert result.succeeded_samples == 1
    by_id = {s.sample_id: s for s in result.sample_scores}
    assert by_id["ok"].error is None
    assert by_id["ok"].scores == {"exact_match": 1.0, "numeric_abs_error": 0.0}
    assert by_id["no-ref"].error and "no reference" in by_id["no-ref"].error
    assert "ValueError" in by_id["bad-number"].error
    # Aggregates use successful samples only; all-failed metric would be null.
    assert result.metrics["exact_match"] == 1.0
    assert result.metrics["numeric_abs_error"] == 0.0


def test_unknown_metric_fails_every_sample_explicitly(service):
    req = make_request(metrics=["nope"])
    r = service.submit(req)
    service.executor.wait(r.task_id, timeout=10)
    status = service.get_status(r.task_id)
    assert status.state is TaskState.SUCCEEDED
    assert status.result.failed_samples == 1
    assert "unknown metric" in status.result.sample_scores[0].error


def test_status_is_explainable_for_nonterminal(service):
    r = service.submit(_slow_request(tenant="explain", n=100, latency=2.0))
    time.sleep(0.1)
    st = service.get_status(r.task_id)
    assert st.state is TaskState.RUNNING
    assert st.started_at is not None and st.finished_at is None
    service.cancel(r.task_id)
    service.executor.wait(r.task_id, timeout=10)

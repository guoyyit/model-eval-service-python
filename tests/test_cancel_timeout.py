"""Cancellation and timeout: terminal states are clean and explainable."""

import threading
import time

from evalsvc.models import TaskStatus

from .conftest import make_request


def slow_request(**overrides):
    samples = [{"id": f"s{i}", "input": "x", "expected": "x"} for i in range(100)]
    return make_request(samples=samples, sample_delay_seconds=0.02, **overrides)


def test_cancel_running_task_leaves_no_partial_result(service):
    task, _ = service.submit(slow_request())
    # wait until it is actually running, then cancel
    deadline = time.monotonic() + 5
    while service.get(task.task_id).status != TaskStatus.RUNNING:
        assert time.monotonic() < deadline
        time.sleep(0.005)
    cancelled = service.cancel(task.task_id)
    assert cancelled.status == TaskStatus.CANCELLED

    final = service.wait_terminal(task.task_id)
    assert final.status == TaskStatus.CANCELLED
    assert final.result is None, "cancelled task must not carry partial results"
    assert final.error == "cancelled by user"


def test_cancel_queued_task(tmp_path):
    """A task still waiting in the queue must cancel cleanly."""
    from evalsvc.service import EvalService

    svc = EvalService(tmp_path / "q.db", max_workers=1, default_quota=100)
    svc.start()
    try:
        blocker, _ = svc.submit(slow_request(model_version="blocker"))
        queued, _ = svc.submit(slow_request(model_version="queued"))
        # the second task cannot start while the single worker is busy
        assert svc.get(queued.task_id).status == TaskStatus.QUEUED
        svc.cancel(queued.task_id)
        final = svc.wait_terminal(queued.task_id)
        assert final.status == TaskStatus.CANCELLED
        assert final.result is None
        svc.cancel(blocker.task_id)
    finally:
        svc.shutdown()


def test_cancel_is_idempotent_and_terminal_state_is_stable(service):
    task, _ = service.submit(slow_request())
    service.cancel(task.task_id)
    again = service.cancel(task.task_id)
    assert again.status == TaskStatus.CANCELLED
    final = service.wait_terminal(task.task_id)
    assert final.status == TaskStatus.CANCELLED
    assert final.result is None


def test_cancel_after_completion_is_a_no_op(service):
    task, _ = service.submit(make_request())
    final = service.wait_terminal(task.task_id)
    assert final.status == TaskStatus.SUCCEEDED
    after = service.cancel(task.task_id)
    assert after.status == TaskStatus.SUCCEEDED
    assert after.result is not None, "completed result must survive a late cancel"


def test_timeout_produces_no_partial_result(service):
    task, _ = service.submit(slow_request(timeout_seconds=0.1))
    final = service.wait_terminal(task.task_id, timeout=15)
    assert final.status == TaskStatus.TIMEOUT
    assert final.result is None, "timed-out task must not carry partial results"
    assert "timeout" in final.error


def test_unfinished_task_status_is_deterministic(service):
    task, _ = service.submit(slow_request())
    try:
        for _ in range(200):
            status = service.get(task.task_id).status
            assert status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
    finally:
        service.cancel(task.task_id)


def test_cancel_races_with_result_write(service):
    """Hammer cancel against near-instant tasks; outcome must be one of
    SUCCEEDED-with-result or CANCELLED-without-result, never mixed."""
    for i in range(30):
        task, _ = service.submit(make_request(model_version=f"race-{i}"))
        service.cancel(task.task_id)
        final = service.wait_terminal(task.task_id)
        assert final.status in {TaskStatus.SUCCEEDED, TaskStatus.CANCELLED}
        if final.status == TaskStatus.SUCCEEDED:
            assert final.result is not None
        else:
            assert final.result is None

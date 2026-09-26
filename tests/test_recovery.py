"""Requirements 5/8: crash recovery semantics."""
from __future__ import annotations

import time

from helpers import make_request, wait_terminal

from app.models import Sample, TaskState
from app.service import EvalService


def test_completed_result_survives_restart(tmp_db_path):
    svc1 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    req = make_request(
        samples=[Sample(id="a", reference="x", prediction="x"),
                 Sample(id="b", reference="y", prediction="y")],
    )
    r = svc1.submit(req)
    wait_terminal(svc1, r.task_id)
    expected = svc1.get_result(r.task_id).model_dump()
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    assert svc2.get_status(r.task_id).state is TaskState.SUCCEEDED
    assert svc2.get_result(r.task_id).model_dump() == expected
    svc2.close()


def test_interrupted_running_task_is_rerun_idempotently(tmp_db_path):
    """Simulate a crash mid-execution: a RUNNING row becomes PENDING on
    reopen, is re-enqueued, and completes exactly once."""
    svc1 = EvalService(db_path=tmp_db_path, autostart=False,
                       enqueue_on_submit=False, default_timeout=30.0)
    req = make_request(idempotency_key="crash")
    r = svc1.submit(req)
    # Mark RUNNING as if the process crashed during evaluation.
    assert svc1.storage.mark_running(r.task_id) is True
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    status = svc2.get_status(r.task_id)
    assert status.state in (TaskState.PENDING, TaskState.RUNNING,
                            TaskState.SUCCEEDED)
    svc2.executor.wait(r.task_id, timeout=10)
    final = svc2.get_status(r.task_id)
    assert final.state is TaskState.SUCCEEDED
    assert final.result.metrics["exact_match"] == 1.0
    # Exactly one task row, no duplicate results/quota.
    assert svc2.storage.tenant_usage("ten") == 1
    # Dedup still maps the fingerprint to the same single task.
    again = svc2.submit(req)
    assert again.task_id == r.task_id and again.reused is True
    svc2.close()


def test_pending_task_re_enqueued_on_restart(tmp_db_path):
    svc1 = EvalService(db_path=tmp_db_path, autostart=False,
                       enqueue_on_submit=False, default_timeout=30.0)
    req = make_request(
        config={"latency": 0.0}, idempotency_key="queued-restart"
    )
    r = svc1.submit(req)
    assert svc1.get_status(r.task_id).state is TaskState.PENDING
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    svc2.executor.wait(r.task_id, timeout=10)
    assert svc2.get_status(r.task_id).state is TaskState.SUCCEEDED
    svc2.close()


def test_cancel_intent_survives_crash_for_running_task(tmp_db_path):
    """If a cancel was requested just before the crash, the RUNNING task is
    finalized CANCELLED on reopen and never re-executed."""
    svc1 = EvalService(db_path=tmp_db_path, autostart=False,
                       enqueue_on_submit=False, default_timeout=30.0)
    req = make_request(config={"latency": 0.01}, idempotency_key="cancel-crash")
    r = svc1.submit(req)
    svc1.storage.mark_running(r.task_id)
    assert svc1.storage.request_cancel(r.task_id, "cancelled") is True
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    # Give the (not re-enqueued) task a moment; it must stay cancelled.
    time.sleep(0.1)
    status = svc2.get_status(r.task_id)
    assert status.state is TaskState.CANCELLED
    assert status.result is None
    svc2.close()


def test_timeout_intent_survives_crash_for_running_task(tmp_db_path):
    svc1 = EvalService(db_path=tmp_db_path, autostart=False,
                       enqueue_on_submit=False, default_timeout=30.0)
    req = make_request(idempotency_key="timeout-crash")
    r = svc1.submit(req)
    svc1.storage.mark_running(r.task_id)
    assert svc1.storage.request_cancel(r.task_id, "timeout") is True
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    status = svc2.get_status(r.task_id)
    assert status.state is TaskState.TIMEOUT
    assert status.result is None
    svc2.close()


def test_recovered_task_produces_deterministic_result(tmp_db_path):
    svc1 = EvalService(db_path=tmp_db_path, autostart=False,
                       enqueue_on_submit=False, default_timeout=30.0)
    req = make_request(
        samples=[Sample(id="a", reference="same", prediction="same")],
        idempotency_key="det",
    )
    r = svc1.submit(req)
    svc1.storage.mark_running(r.task_id)
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    svc2.executor.wait(r.task_id, timeout=10)
    recovered = svc2.get_result(r.task_id).model_dump()
    svc2.close()

    svc3 = EvalService(db_path=tmp_db_path, default_timeout=30.0)
    assert svc3.get_result(r.task_id).model_dump() == recovered
    svc3.close()

"""Crash recovery: completed results survive; in-flight tasks resume."""

import os

from evalsvc.models import TaskStatus
from evalsvc.service import EvalService

from .conftest import make_request


def _restart(db_path, **kwargs) -> EvalService:
    svc = EvalService(db_path, **kwargs)
    svc.start()
    return svc


def test_completed_results_survive_restart(tmp_path):
    db = tmp_path / "eval.db"
    svc = _restart(db)
    task, _ = svc.submit(make_request())
    final = svc.wait_terminal(task.task_id)
    assert final.status == TaskStatus.SUCCEEDED
    expected_json = final.result.model_dump_json()
    svc.shutdown()

    svc2 = _restart(db)
    try:
        restored = svc2.get(task.task_id)
        assert restored is not None
        assert restored.status == TaskStatus.SUCCEEDED
        assert restored.result.model_dump_json() == expected_json
    finally:
        svc2.shutdown()


def test_inflight_task_is_reexecuted_after_crash(tmp_path):
    db = tmp_path / "eval.db"
    svc = _restart(db, max_workers=1)
    slow = make_request(
        model_version="slow",
        sample_delay_seconds=0.5,
        samples=[{"id": "a", "input": "x", "expected": "x"}],
    )
    task, _ = svc.submit(slow)
    # simulate a crash: do not wait for workers, do not mark anything
    svc.shutdown(wait=False)

    svc2 = _restart(db, max_workers=1)
    try:
        recovered = svc2.get(task.task_id)
        assert recovered.status in {TaskStatus.QUEUED, TaskStatus.RUNNING}
        final = svc2.wait_terminal(task.task_id, timeout=15)
        assert final.status == TaskStatus.SUCCEEDED
        assert final.result is not None
        # idempotent: exactly one task row and one quota charge
        assert len(svc2.list("t1")) == 1
        assert svc2.quota.usage("t1") == 1
    finally:
        svc2.shutdown()


def test_quota_is_not_double_charged_across_restart(tmp_path):
    db = tmp_path / "eval.db"
    svc = _restart(db)
    for i in range(3):
        svc.submit(make_request(model_version=f"v{i}"))
    assert svc.quota.usage("t1") == 3
    svc.shutdown(wait=False)

    svc2 = _restart(db)
    try:
        assert svc2.quota.usage("t1") == 3, "recovery must not re-charge quota"
        for t in svc2.list("t1"):
            svc2.wait_terminal(t.task_id, timeout=15)
        assert svc2.quota.usage("t1") == 3
        # resubmitting the same request after recovery hits the cache
        again, created = svc2.submit(make_request(model_version="v0"))
        assert created is False
        assert svc2.quota.usage("t1") == 3
    finally:
        svc2.shutdown()


def test_cancelled_task_stays_cancelled_across_restart(tmp_path):
    db = tmp_path / "eval.db"
    svc = _restart(db, max_workers=1)
    slow = make_request(
        model_version="slow-cancel",
        sample_delay_seconds=0.5,
        samples=[{"id": "a", "input": "x", "expected": "x"}],
    )
    task, _ = svc.submit(slow)
    svc.cancel(task.task_id)
    assert svc.wait_terminal(task.task_id).status == TaskStatus.CANCELLED
    svc.shutdown(wait=False)

    svc2 = _restart(db)
    try:
        restored = svc2.get(task.task_id)
        assert restored.status == TaskStatus.CANCELLED
        assert restored.result is None
    finally:
        svc2.shutdown()

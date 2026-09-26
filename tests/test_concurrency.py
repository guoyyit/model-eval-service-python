"""Requirements 6/8: bounded concurrency and slow-task non-blocking."""
from __future__ import annotations

import threading
import time

from helpers import make_request

from app.models import Sample, TaskState
from app.service import EvalService


def _slow(tenant, key, latency=2.0, n=100):
    return make_request(
        tenant_id=tenant, idempotency_key=key,
        samples=[Sample(id=str(i), reference="r") for i in range(n)],
        config={"latency": latency},
    )


def _fast(tenant, key):
    return make_request(tenant_id=tenant, idempotency_key=key)


def test_bounded_concurrency_runs_at_most_max_workers(tmp_db_path):
    max_workers = 3
    svc = EvalService(db_path=tmp_db_path, max_workers=max_workers,
                      default_timeout=None)
    active = 0
    peak = 0
    lock = threading.Lock()

    from app.evaluator import register_backend

    class CountingBackend:
        def predict(self, model_version, text, config):
            nonlocal active, peak
            with lock:
                active += 1
                peak = max(peak, active)
            time.sleep(0.05)
            with lock:
                active -= 1
            return "same"

    register_backend("counter", CountingBackend())
    try:
        reqs = [
            make_request(tenant_id="bounded", idempotency_key=f"k{i}",
                         samples=[Sample(id="0", reference="same")],
                         config={"backend": "counter"})
            for i in range(9)
        ]
        ids = [svc.submit(r).task_id for r in reqs]
        for tid in ids:
            svc.executor.wait(tid, timeout=20)
        for tid in ids:
            assert svc.get_status(tid).state is TaskState.SUCCEEDED
        assert peak <= max_workers, peak
        assert peak == max_workers, peak
    finally:
        svc.close()


def test_slow_task_does_not_block_fast_task(tmp_db_path):
    svc = EvalService(db_path=tmp_db_path, max_workers=2, default_timeout=None)
    # Occupy both workers with slow tasks.
    slow_ids = [svc.submit(_slow("mix", f"s{i}")).task_id for i in range(2)]
    time.sleep(0.1)
    # Queue a fast task; as soon as a slow task is cancelled it must run.
    fast_id = svc.submit(_fast("mix", "fast")).task_id
    assert svc.get_status(fast_id).state is TaskState.PENDING
    for tid in slow_ids:
        svc.cancel(tid)
    svc.executor.wait(fast_id, timeout=10)
    assert svc.get_status(fast_id).state is TaskState.SUCCEEDED
    svc.close()


def test_many_independent_tenants_progress_concurrently(tmp_db_path):
    svc = EvalService(db_path=tmp_db_path, max_workers=4, default_timeout=None)
    # Mild latency but all tasks should finish well within serial*workers time.
    reqs = [
        make_request(tenant_id=f"tenant{i}",
                     samples=[Sample(id="0", reference="same", prediction="same")],
                     config={"latency": 0.05})
        for i in range(8)
    ]
    start = time.monotonic()
    ids = [svc.submit(r).task_id for r in reqs]
    for tid in ids:
        svc.executor.wait(tid, timeout=20)
    elapsed = time.monotonic() - start
    for tid in ids:
        assert svc.get_status(tid).state is TaskState.SUCCEEDED
    # 8 tasks x 0.05 with 4 workers ~ 0.1; serial would be 0.4.
    assert elapsed < 0.35, elapsed
    svc.close()

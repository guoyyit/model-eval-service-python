"""Concurrency: bounded parallelism, no head-of-line blocking, isolation."""

import threading
import time

import pytest

from evalsvc.models import TaskStatus
from evalsvc.service import EvalService
from evalsvc.store import QuotaExceeded

from .conftest import make_request


def test_slow_task_does_not_block_fast_tasks(tmp_path):
    svc = EvalService(tmp_path / "c.db", max_workers=2, default_quota=100)
    svc.start()
    try:
        slow, _ = svc.submit(make_request(
            model_version="slow",
            sample_delay_seconds=0.05,
            samples=[{"id": f"s{i}", "input": "x", "expected": "x"}
                     for i in range(40)],
        ))
        fast_tasks = [
            svc.submit(make_request(model_version=f"fast-{i}"))[0]
            for i in range(4)
        ]
        start = time.monotonic()
        for t in fast_tasks:
            final = svc.wait_terminal(t.task_id, timeout=10)
            assert final.status == TaskStatus.SUCCEEDED
        fast_elapsed = time.monotonic() - start
        # slow task needs ~2s; fast tasks must finish well before that
        assert fast_elapsed < 1.5, f"fast tasks blocked for {fast_elapsed:.2f}s"
        assert svc.get(slow.task_id).status in {
            TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.SUCCEEDED,
        }
        svc.cancel(slow.task_id)
    finally:
        svc.shutdown()


def test_concurrency_is_bounded(tmp_path):
    svc = EvalService(tmp_path / "b.db", max_workers=2, default_quota=100)
    svc.start()
    try:
        running_peak = 0
        lock = threading.Lock()
        stop = threading.Event()

        def observe():
            nonlocal running_peak
            while not stop.is_set():
                running = sum(
                    1 for t in svc.list() if t.status == TaskStatus.RUNNING
                )
                with lock:
                    running_peak = max(running_peak, running)
                time.sleep(0.005)

        observer = threading.Thread(target=observe)
        observer.start()
        tasks = [
            svc.submit(make_request(
                model_version=f"m{i}",
                sample_delay_seconds=0.05,
                samples=[{"id": "a", "input": "x", "expected": "x"}],
            ))[0]
            for i in range(6)
        ]
        for t in tasks:
            svc.wait_terminal(t.task_id, timeout=15)
        stop.set()
        observer.join()
        assert running_peak <= 2, f"peak concurrency {running_peak} exceeded bound"
        assert running_peak >= 1
        assert all(
            svc.get(t.task_id).status == TaskStatus.SUCCEEDED for t in tasks
        )
    finally:
        svc.shutdown()


def test_quota_exhaustion_rejects_new_but_keeps_existing(tmp_path):
    svc = EvalService(tmp_path / "q.db", max_workers=2, default_quota=2)
    svc.start()
    try:
        svc.submit(make_request(model_version="a", tenant="t1"))
        svc.submit(make_request(model_version="b", tenant="t1"))
        with pytest.raises(QuotaExceeded):
            svc.submit(make_request(model_version="c", tenant="t1"))
        # rejected submission leaves no task behind
        assert len(svc.list("t1")) == 2

        # tightening the limit does not affect already-accepted tasks
        svc.quota.set_limit("t1", 0)
        for t in svc.list("t1"):
            final = svc.wait_terminal(t.task_id, timeout=15)
            assert final.status == TaskStatus.SUCCEEDED
        # but new submissions are now rejected
        with pytest.raises(QuotaExceeded):
            svc.submit(make_request(model_version="d", tenant="t1"))
    finally:
        svc.shutdown()


def test_quotas_are_isolated_per_tenant(tmp_path):
    svc = EvalService(tmp_path / "i.db", max_workers=2, default_quota=1)
    svc.start()
    try:
        svc.submit(make_request(tenant="rich"))
        svc.submit(make_request(tenant="poor"))
        with pytest.raises(QuotaExceeded):
            svc.submit(make_request(tenant="poor", model_version="other"))
        # raising poor's limit allows more submissions immediately
        svc.quota.set_limit("poor", 3)
        svc.submit(make_request(tenant="poor", model_version="other"))
        assert svc.quota.usage("rich") == 1
        assert svc.quota.usage("poor") == 2
    finally:
        svc.shutdown()


def test_task_listing_is_isolated_per_tenant(service):
    service.submit(make_request(tenant="alpha", model_version="m1"))
    service.submit(make_request(tenant="alpha", model_version="m2"))
    service.submit(make_request(tenant="beta", model_version="m1"))
    assert len(service.list("alpha")) == 2
    assert len(service.list("beta")) == 1
    assert len(service.list()) == 3

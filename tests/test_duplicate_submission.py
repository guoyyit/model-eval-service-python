"""Duplicate submission: one execution, one quota charge, stable task id."""

import threading

from evalsvc.models import TaskStatus

from .conftest import make_request


def test_sequential_duplicates_return_same_task(service):
    req = make_request()
    first, created1 = service.submit(req)
    second, created2 = service.submit(req)
    assert created1 is True
    assert created2 is False
    assert first.task_id == second.task_id
    assert service.quota.usage("t1") == 1


def test_concurrent_duplicates_create_single_task(service):
    req = make_request()
    results: list[tuple[str, bool]] = []
    lock = threading.Lock()
    barrier = threading.Barrier(8)

    def submit():
        barrier.wait()
        record, created = service.submit(req)
        with lock:
            results.append((record.task_id, created))

    threads = [threading.Thread(target=submit) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    task_ids = {tid for tid, _ in results}
    assert len(task_ids) == 1, f"expected one task, got {task_ids}"
    assert sum(1 for _, created in results if created) == 1
    assert service.quota.usage("t1") == 1


def test_success_result_is_cached_for_identical_request(service):
    req = make_request()
    first, _ = service.submit(req)
    final = service.wait_terminal(first.task_id)
    assert final.status == TaskStatus.SUCCEEDED

    again, created = service.submit(req)
    assert created is False
    assert again.task_id == first.task_id
    assert again.status == TaskStatus.SUCCEEDED
    assert again.result is not None
    assert service.quota.usage("t1") == 1


def test_same_request_different_tenants_are_isolated(service):
    req_a = make_request(tenant="tenant-a")
    req_b = make_request(tenant="tenant-b")
    task_a, _ = service.submit(req_a)
    task_b, _ = service.submit(req_b)
    assert task_a.task_id != task_b.task_id
    assert service.quota.usage("tenant-a") == 1
    assert service.quota.usage("tenant-b") == 1

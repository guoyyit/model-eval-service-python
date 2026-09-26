"""Requirement 1/8: duplicate submission dedup and single quota charge."""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from helpers import make_request, wait_terminal

from app.models import TaskState


def test_identical_request_returns_same_task_id(service):
    req = make_request()
    r1 = service.submit(req)
    r2 = service.submit(req)
    assert r1.task_id == r2.task_id
    assert r1.reused is False
    assert r2.reused is True


def test_duplicate_concurrent_submission_creates_one(service):
    # Slow enough that many threads overlap in the submission transaction.
    req = make_request(config={"latency": 0.05})
    responses = []
    lock = threading.Lock()

    def submit():
        with lock:
            responses.append(service.submit(req))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(lambda _: submit(), range(40)))

    assert len({r.task_id for r in responses}) == 1
    assert sum(1 for r in responses if not r.reused) == 1
    tid = responses[0].task_id
    status = wait_terminal(service, tid)
    assert status.state is TaskState.SUCCEEDED
    # Quota charged exactly once despite 40 submissions.
    assert service.storage.tenant_usage("ten") == 1


def test_completed_task_dedup_returns_same_readable_result(service):
    req = make_request()
    r1 = service.submit(req)
    wait_terminal(service, r1.task_id)
    result1 = service.get_result(r1.task_id)

    r2 = service.submit(req)
    assert r2.reused is True
    result2 = service.get_result(r2.task_id)
    assert result1 == result2
    assert service.storage.tenant_usage("ten") == 1


def test_idempotency_key_partitions_dedup(service):
    r_a = service.submit(
        make_request(idempotency_key="key-a")
    )
    r_b = service.submit(
        make_request(idempotency_key="key-b")
    )
    r_c = service.submit(
        make_request(idempotency_key="key-a")
    )
    assert r_a.task_id != r_b.task_id
    assert r_a.task_id == r_c.task_id
    assert r_c.reused is True


def test_concurrent_distinct_submissions_respect_quota(service):
    service.set_quota("cq", None, 10)
    import threading
    from concurrent.futures import ThreadPoolExecutor

    outcomes = []
    lock = threading.Lock()

    def one(i):
        try:
            r = service.submit(make_request(tenant_id="cq",
                                            idempotency_key=f"k{i}"))
            with lock:
                outcomes.append(("ok", r.task_id))
        except Exception as exc:
            with lock:
                outcomes.append(("rejected", type(exc).__name__))

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(one, range(30)))

    accepted = [o for o in outcomes if o[0] == "ok"]
    rejected = [o for o in outcomes if o[0] == "rejected"]
    assert len(accepted) == 10
    assert len(rejected) == 20
    assert len({tid for _, tid in accepted}) == 10
    assert service.storage.tenant_usage("cq") == 10

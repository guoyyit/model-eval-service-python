"""Requirements 4/8: tenant/project isolation, quota exhaustion, and
immunity of existing tasks to later limit changes."""
from __future__ import annotations

import pytest
from helpers import make_request

from app.models import Sample, TaskState
from app.storage import QuotaExceededError


def test_tasks_isolated_between_tenants(service):
    r1 = service.submit(make_request(tenant_id="alpha", idempotency_key="k"))
    r2 = service.submit(make_request(tenant_id="beta", idempotency_key="k"))
    assert r1.task_id != r2.task_id
    status = service.get_status(r1.task_id)
    assert status.tenant_id == "alpha"
    # Rows are owned by their tenant; HTTP layer enforces cross-tenant 404
    # (covered in the API test suite).
    assert service.storage.get_task(r1.task_id)["tenant_id"] == "alpha"


def test_project_isolation_under_same_tenant(service):
    ra = service.submit(make_request(tenant_id="t", project_id="p1",
                                     idempotency_key="k"))
    rb = service.submit(make_request(tenant_id="t", project_id="p2",
                                     idempotency_key="k"))
    assert ra.task_id != rb.task_id
    tasks_p1 = service.list_tasks("t", "p1")
    assert [x.task_id for x in tasks_p1] == [ra.task_id]
    tasks_all = service.list_tasks("t")
    assert {x.task_id for x in tasks_all} == {ra.task_id, rb.task_id}


def test_quota_exhaustion_rejects_new_submission(service):
    service.set_quota("limited", None, 1)
    service.submit(make_request(tenant_id="limited", idempotency_key="a"))
    with pytest.raises(QuotaExceededError) as exc:
        service.submit(make_request(tenant_id="limited", idempotency_key="b"))
    assert exc.value.scope == "limited"
    # Re-submitting the same request still works (no extra charge).
    again = service.submit(make_request(tenant_id="limited",
                                        idempotency_key="a"))
    assert again.reused is True


def test_project_level_quota(service):
    service.set_quota("t", "p", 1)
    service.submit(make_request(tenant_id="t", project_id="p",
                                idempotency_key="a"))
    with pytest.raises(QuotaExceededError) as exc:
        service.submit(make_request(tenant_id="t", project_id="p",
                                    idempotency_key="b"))
    assert exc.value.scope == "t:p"
    # A different project under the same tenant is unaffected.
    other = service.submit(make_request(tenant_id="t", project_id="other",
                                        idempotency_key="a"))
    assert other.reused is False


def test_tenant_quota_covers_projects(service):
    service.set_quota("t", None, 1)
    service.submit(make_request(tenant_id="t", project_id="p1",
                                idempotency_key="a"))
    with pytest.raises(QuotaExceededError) as exc:
        service.submit(make_request(tenant_id="t", project_id="p2",
                                    idempotency_key="b"))
    assert exc.value.scope == "t"


def test_quota_change_does_not_affect_existing_tasks(service):
    # Plenty of room initially; submit a long-running task.
    slow = make_request(
        tenant_id="churn",
        samples=[Sample(id=str(i), reference="r") for i in range(20)],
        config={"latency": 0.02},
    )
    r = service.submit(slow)
    # Now tighten the quota to zero: running task must continue to succeed.
    service.set_quota("churn", None, 0)
    info = service.get_quota("churn")
    assert info.limit == 0 and info.used == 1
    service.executor.wait(r.task_id, timeout=10)
    assert service.get_status(r.task_id).state is TaskState.SUCCEEDED
    # New work is rejected.
    with pytest.raises(QuotaExceededError):
        service.submit(make_request(
            tenant_id="churn", idempotency_key="new"))
    # Raising again admits new work without touching the old task.
    service.set_quota("churn", None, 10)
    r2 = service.submit(make_request(tenant_id="churn", idempotency_key="new"))
    assert r2.reused is False


def test_quota_not_double_charged_after_restart(tmp_db_path):
    from app.service import EvalService
    svc1 = EvalService(db_path=tmp_db_path, default_quota=100)
    svc1.set_quota("ten", None, 5)
    svc1.submit(make_request(tenant_id="ten", idempotency_key="a"))
    svc1.submit(make_request(tenant_id="ten", idempotency_key="b"))
    svc1.close()

    svc2 = EvalService(db_path=tmp_db_path, default_quota=100)
    assert svc2.storage.tenant_usage("ten") == 2
    # Still room for exactly 3 more tasks (not 5).
    svc2.submit(make_request(tenant_id="ten", idempotency_key="c"))
    svc2.submit(make_request(tenant_id="ten", idempotency_key="d"))
    svc2.submit(make_request(tenant_id="ten", idempotency_key="e"))
    with pytest.raises(QuotaExceededError):
        svc2.submit(make_request(tenant_id="ten", idempotency_key="f"))
    svc2.close()

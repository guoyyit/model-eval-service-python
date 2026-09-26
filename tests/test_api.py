"""Requirement 8: HTTP API behaviour incl. isolation and error codes."""
from __future__ import annotations

import time


def _body(**overrides):
    body = {
        "tenant_id": "ten",
        "model_version": "v1",
        "samples": [{"id": "s1", "reference": "hello", "prediction": "hello"}],
        "metrics": ["exact_match"],
    }
    body.update(overrides)
    return body


def _submit(client, **overrides):
    return client.post("/api/v1/evaluations", json=_body(**overrides))


def _wait(client, tid, tenant="ten", timeout=10.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = client.get(f"/api/v1/evaluations/{tid}?tenant_id={tenant}")
        state = r.json()["state"]
        if state in ("succeeded", "failed", "cancelled", "timeout"):
            return r
        time.sleep(0.02)
    raise AssertionError("task did not finish in time")


def test_submit_and_poll(client):
    r = _submit(client)
    assert r.status_code == 201
    tid = r.json()["task_id"]
    assert r.json()["reused"] is False
    final = _wait(client, tid)
    assert final.json()["state"] == "succeeded"
    res = client.get(f"/api/v1/evaluations/{tid}/result?tenant_id=ten")
    assert res.status_code == 200
    assert res.json()["metrics"] == {"exact_match": 1.0}


def test_duplicate_submit_same_id_no_extra_quota(client, service):
    r1 = _submit(client)
    r2 = _submit(client)
    assert r1.json()["task_id"] == r2.json()["task_id"]
    assert r2.json()["reused"] is True
    assert service.storage.tenant_usage("ten") == 1


def test_cross_tenant_get_returns_404(client):
    tid = _submit(client).json()["task_id"]
    _wait(client, tid)
    r = client.get(f"/api/v1/evaluations/{tid}?tenant_id=other")
    assert r.status_code == 404
    r = client.get(f"/api/v1/evaluations/{tid}/result?tenant_id=other")
    assert r.status_code == 404
    r = client.post(
        f"/api/v1/evaluations/{tid}/cancel?tenant_id=other")
    assert r.status_code == 404


def test_quota_exhaustion_returns_429(client, service):
    service.set_quota("ten", None, 1)
    _submit(client, idempotency_key="a")
    r = _submit(client, idempotency_key="b")
    assert r.status_code == 429
    detail = r.json()["detail"]
    assert detail["error"] == "quota_exceeded"
    assert detail["used"] == 1 and detail["limit"] == 1
    # Duplicate still served.
    dup = _submit(client, idempotency_key="a")
    assert dup.status_code == 201 and dup.json()["reused"] is True


def test_cancel_endpoint(client, service):
    body = _body(
        samples=[{"id": str(i), "reference": "r"} for i in range(50)],
        config={"latency": 0.05},
    )
    r = client.post("/api/v1/evaluations", json=body)
    tid = r.json()["task_id"]
    time.sleep(0.1)
    cr = client.post(f"/api/v1/evaluations/{tid}/cancel?tenant_id=ten")
    assert cr.json()["state"] in ("running", "cancelled")
    final = _wait(client, tid)
    assert final.json()["state"] == "cancelled"
    assert final.json()["result"] is None


def test_timeout_end_state(client):
    body = _body(
        samples=[{"id": str(i), "reference": "r"} for i in range(10)],
        config={"latency": 0.1}, timeout_seconds=0.15,
    )
    r = client.post("/api/v1/evaluations", json=body)
    tid = r.json()["task_id"]
    final = _wait(client, tid)
    assert final.json()["state"] == "timeout"


def test_sample_level_failures_endpoint(client):
    body = _body(samples=[
        {"id": "ok", "reference": "x", "prediction": "x"},
        {"id": "bad", "reference": None, "prediction": "p"},
    ])
    r = client.post("/api/v1/evaluations", json=body)
    tid = r.json()["task_id"]
    _wait(client, tid)
    scores = client.get(
        f"/api/v1/evaluations/{tid}/samples?tenant_id=ten").json()
    by_id = {s["sample_id"]: s for s in scores}
    assert by_id["ok"]["error"] is None
    assert "no reference" in by_id["bad"]["error"]


def test_list_tenant_evaluations_and_state_filter(client):
    _submit(client, idempotency_key="a")
    _submit(client, idempotency_key="b")
    r = client.get("/api/v1/tenants/ten/evaluations")
    assert r.status_code == 200 and len(r.json()) == 2
    r_ok = client.get("/api/v1/tenants/ten/evaluations?state=succeeded")
    assert len(r_ok.json()) == 2
    assert client.get(
        "/api/v1/tenants/ten/evaluations?state=bogus").status_code == 400
    # Other tenant sees nothing.
    assert client.get(
        "/api/v1/tenants/other/evaluations").json() == []


def test_quota_admin_endpoints(client):
    r = client.put("/api/v1/quotas",
                   json={"tenant_id": "ten", "project_id": "p", "limit": 7})
    assert r.status_code == 200
    info = r.json()
    assert info["scope"] == "ten:p" and info["limit"] == 7 and info["used"] == 0
    listing = client.get("/api/v1/quotas").json()
    assert any(q["scope"] == "ten:p" for q in listing)


def test_validation_errors(client):
    r = client.post("/api/v1/evaluations",
                    json=_body(samples=[]))
    assert r.status_code == 422
    r = client.post("/api/v1/evaluations",
                    json=_body(tenant_id=""))
    assert r.status_code == 422
    r = client.post("/api/v1/evaluations",
                    json=_body(timeout_seconds=0))
    assert r.status_code == 422
    r = client.post("/api/v1/evaluations",
                    json=_body(samples=[
                        {"id": "x", "reference": "a", "prediction": "a"},
                        {"id": "x", "reference": "b", "prediction": "b"},
                    ]))
    assert r.status_code == 422
    r = client.post("/api/v1/evaluations",
                    json=_body(metrics=[]))
    assert r.status_code == 422


def test_metrics_catalog(client):
    r = client.get("/api/v1/metrics")
    assert r.status_code == 200
    assert set(r.json()["metrics"]) == {
        "exact_match", "token_f1", "numeric_abs_error"}
